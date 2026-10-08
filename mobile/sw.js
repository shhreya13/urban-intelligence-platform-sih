/* Urban Intelligence mobile app -- service worker.
   Caches ONLY the app shell so the app opens instantly and installs.
   Every API call, evidence image and recording goes to the network:
   event data is never served from a cache, so a status change made on
   another device is always what this device shows. */
const SHELL = "uip-mobile-v1";
const FILES = ["./", "./index.html", "./manifest.webmanifest", "./icon.svg", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== SHELL).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || !url.pathname.startsWith("/mobile/")) return;
  // network first, shell cache only when offline
  e.respondWith(fetch(e.request).then(res => {
    const copy = res.clone();
    caches.open(SHELL).then(c => c.put(e.request, copy));
    return res;
  }).catch(() => caches.match(e.request).then(r => r || caches.match("./index.html"))));
});
