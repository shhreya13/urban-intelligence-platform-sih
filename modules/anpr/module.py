"""
MODULE 5 -- ANPR / NUMBER PLATE                 (owner: Person 5)
=================================================================

Wraps **Person 5's ANPRPipeline** exactly as they published it, loaded
unmodified from ``vendor/person5_anpr/anpr_pipeline.py``:

    plate detector (YOLO) -> plate crop -> enhancement -> EasyOCR
    -> multi-frame voting -> format validation -> plate + confidence

The single entry point of their module is
``ANPRPipeline.process_incident(track_id, vehicle_frames)`` and that is the
only thing this adapter calls.  Their OCR, their voting, their correction
tables -- untouched.

**ANPR NEVER RUNS PER FRAME.**  ``process()`` returns an empty list, always.
The pipeline only reaches this module through ``identify()``, which the
orchestrator calls when an incident event carries
``metadata.requires_identification``.  Normal driving never gets here, which
is exactly the compute budget the Nano phase will depend on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.event import Event, EventType, Severity, SourceModule, make_event
from core.frame_context import FrameContext
from core.module_base import BaseModule, ModuleUnavailable
from core.vendor_loader import load_vendor_module, VendorImportError

ANPR_SRC = "person5_anpr/anpr_pipeline.py"


class ANPRModule(BaseModule):
    name = "anpr"
    owner = "Person 5"

    def __init__(self, config, evidence=None):
        super().__init__(config.section("modules.anpr"))
        self.platform_config = config
        self.evidence = evidence

        self.pipeline = None
        self.trigger_event_types = set(
            self.config.get(
                "trigger_event_types",
                ["hit_and_run", "collision", "rash_driving"],
            )
        )
        self.min_trigger_severity = str(
            self.config.get("min_trigger_severity", "HIGH")
        ).upper()
        self.save_evidence = bool(self.config.get("save_evidence", True))
        self.max_identifications = int(self.config.get("max_identifications", 40))

        self._identified: Dict[int, Dict[str, Any]] = {}
        self.trigger_count = 0
        self.success_count = 0

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        model_path = self.config.get("model") or "models/plate_best.pt"
        path = self.platform_config.resolve(model_path)
        if not path.exists():
            raise ModuleUnavailable(f"plate detector weights not found: {path}")

        try:
            anpr_module = load_vendor_module(ANPR_SRC)
        except VendorImportError as exc:
            # Almost always a missing easyocr install.  ANPR going missing must
            # never take the pipeline down -- incidents are still reported,
            # just without a registration number.
            raise ModuleUnavailable(
                f"Person 5 ANPR pipeline could not be imported ({exc}). "
                "Install it with: pip install easyocr"
            ) from exc

        # Respect the configured OCR budget without editing their file.
        max_frames = self.config.get("max_frames_to_ocr")
        if max_frames:
            anpr_module.MAX_FRAMES_TO_OCR = int(max_frames)
        min_sharpness = self.config.get("min_sharpness")
        if min_sharpness is not None:
            anpr_module.MIN_SHARPNESS = float(min_sharpness)

        self.log.info("loading Person 5 ANPRPipeline (%s) ...", path.name)
        self.pipeline = anpr_module.ANPRPipeline(
            plate_model_path=str(path),
            gpu=bool(self.config.get("gpu", False)),
        )
        self.log.info(
            "ANPR ready -- event-triggered only (triggers: %s, min severity %s)",
            sorted(self.trigger_event_types), self.min_trigger_severity,
        )

    # -- per frame ----------------------------------------------------------

    def process(self, ctx: FrameContext) -> List[Event]:
        """Intentionally empty.  ANPR is event-triggered, never continuous."""
        return []

    # -- trigger ------------------------------------------------------------

    def should_trigger(self, event: Event) -> bool:
        if self.status != "READY":
            return False
        if event.event_type not in self.trigger_event_types:
            return False
        if not event.metadata.get("requires_identification"):
            return False
        if Severity.rank(event.severity) < Severity.rank(self.min_trigger_severity):
            return False
        track_id = event.metadata.get("anpr_track_id", event.track_id)
        if track_id is None:
            return False
        if track_id in self._identified:
            return False
        if len(self._identified) >= self.max_identifications:
            self.log.warning(
                "identification budget of %s reached -- skipping track %s",
                self.max_identifications, track_id,
            )
            return False
        return True

    def identify(
        self,
        source_event: Event,
        crops: List[Any],
        ctx: Optional[FrameContext] = None,
    ) -> Optional[Event]:
        """Run Person 5's pipeline for one flagged vehicle.

        ``crops`` are the buffered vehicle crops for that track id, taken
        from the shared crop buffer -- i.e. frames from *before* the incident
        as well as after it.
        """
        track_id = source_event.metadata.get("anpr_track_id", source_event.track_id)
        if track_id is None or self.pipeline is None:
            return None

        self.trigger_count += 1
        if not crops:
            self.log.warning(
                "no buffered crops for track %s -- cannot identify", track_id
            )
            self._identified[track_id] = {"plate": "", "confidence": 0.0}
            return self._build_event(source_event, ctx, track_id, {
                "plate": "", "confidence": 0.0, "valid": False, "frames_used": 0,
                "note": "no buffered vehicle crops were available for this track",
            })

        self.log.info(
            "ANPR triggered by %s on track %s (%s buffered crops)",
            source_event.event_type, track_id, len(crops),
        )
        try:
            result = self.pipeline.process_incident(
                track_id=track_id, vehicle_frames=crops
            )
        except Exception as exc:                                 # noqa: BLE001
            self.error_count += 1
            self.log.error("ANPR failed for track %s: %s", track_id, exc)
            return None

        self._identified[track_id] = result
        if result.get("plate"):
            self.success_count += 1
        return self._build_event(source_event, ctx, track_id, result)

    # -- helpers ------------------------------------------------------------

    def _build_event(self, source_event, ctx, track_id, result) -> Event:
        evidence_image = None
        if self.save_evidence and self.evidence is not None and ctx is not None:
            best = result.get("_best_crop")
            if best is not None:
                evidence_image = self.evidence.save_image(
                    best, "anpr", ctx.bus_id, ctx.frame_number
                )
        if evidence_image is None:
            evidence_image = source_event.evidence_image

        plate = result.get("plate") or ""
        confidence = float(result.get("confidence", 0.0) or 0.0)
        note = result.get("note")

        # SPEC 13: a poor read is never published as a confirmed plate, and
        # we never invent one. But if OCR did read *something*, show that
        # real (if shaky) text -- clearly marked as unverified -- rather
        # than discarding it as a bare UNKNOWN. The crop is still saved so
        # a human can read it off the evidence image either way.
        minimum = float(self.config.get("min_plate_confidence", 0.45) or 0.0)
        unverified = False
        if plate and confidence < minimum:
            unverified = True
            note = (
                f"OCR read '{plate}' at {confidence:.2f}, below the "
                f"{minimum:.2f} confirmation threshold -- shown as an "
                f"unverified partial read, not a confirmed plate"
            )
            self.log.info(
                "[ANPR] track %s: low-confidence read %.2f -> unverified partial (%s)",
                track_id, confidence, plate,
            )
            result = {**result, "low_confidence_text": plate}

        if plate:
            # Indian plates run up to 10 chars (SS NN LL NNNN, no spaces).
            # Pad the real detected characters with underscores out to that
            # length -- this reads as "positions not read", never as digits
            # or letters that could be mistaken for an actual detection.
            PLATE_LENGTH = 10
            pad = max(0, PLATE_LENGTH - len(plate))
            registration = f"{plate}{'_' * pad}" if unverified else plate
        else:
            registration = "UNKNOWN"

        if plate and not unverified and result.get("valid"):
            severity = Severity.HIGH
        elif plate and not unverified:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        event = make_event(
            EventType.VEHICLE_IDENTIFICATION,
            SourceModule.ANPR,
            confidence=round(confidence, 3),
            severity=severity,
            class_name=source_event.class_name,
            track_id=track_id,
            bounding_box=source_event.bounding_box,
            evidence_image=evidence_image,
            gps=source_event.gps,
            bus_id=source_event.bus_id,
            camera_id=source_event.camera_id,
            road_segment=source_event.road_segment,
            metadata={
                "vehicle_track_id": track_id,
                "registration_number": registration,
                "unverified_partial_read": unverified,
                "plate_confidence": round(confidence, 3),
                "plate_valid_format": bool(result.get("valid")),
                "frames_used": result.get("frames_used", 0),
                "votes": result.get("votes", {}),
                "raw_candidates": [
                    [text, round(float(conf), 3)]
                    for text, conf in (result.get("raw_candidates") or [])
                ],
                "note": note,
                "low_confidence_text": result.get("low_confidence_text"),
                "min_plate_confidence": minimum,
                "triggered_by_event_id": source_event.event_id,
                "triggered_by_event_type": source_event.event_type,
                "engine": "person5_anpr/anpr_pipeline.py (EasyOCR + plate YOLO)",
            },
        )
        # Let the source incident carry the plate too, so a single row on the
        # dashboard shows "hit and run -- TN12AB1234".
        source_event.metadata["registration_number"] = registration
        source_event.metadata["plate_confidence"] = round(confidence, 3)
        source_event.metadata["identification_event_id"] = event.event_id
        return event

    def known_plate(self, track_id: int) -> Optional[str]:
        record = self._identified.get(track_id)
        return record.get("plate") if record else None

    def stats(self) -> Dict[str, Any]:
        data = super().stats()
        data["triggers"] = self.trigger_count
        data["plates_read"] = self.success_count
        data["tracks_identified"] = len(self._identified)
        return data
