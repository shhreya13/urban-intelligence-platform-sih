import cv2


def main():
    print("Starting laptop camera...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open laptop camera.")
        print("Try changing camera source from 0 to 1.")
        return

    print("Camera opened successfully.")
    print("Press 'q' to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame from camera.")
            break

        cv2.imshow("Traffic Intelligence - Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Camera closed.")


if __name__ == "__main__":
    main()