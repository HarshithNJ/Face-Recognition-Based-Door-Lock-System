import face_recognition
import os
import cv2
import numpy as np
import math
import sys
import serial
from serial.tools.list_ports import comports
from time import time

# Helper function to calculate face confidence
def face_confidence(face_distance, face_match_threshold=0.6):
    range_ = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_ * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.confirmation_received = False

        self.encode_faces()

        # List available serial ports
        self.serial_ports = [port.device for port in comports()]

        # Prompt user to select a serial port
        print("Available serial ports:")
        for i, port in enumerate(self.serial_ports):
            print(f"{i+1}. {port}")
        port_index = int(input("Enter the index of the serial port you want to use: ")) - 1
        self.selected_port = self.serial_ports[port_index]

        # Initialize serial connection with selected Arduino
        self.arduino = serial.Serial(self.selected_port, 9600, timeout=1)

    # Batch encode faces
    def encode_faces(self):
        start_time = time()
        for image_name in os.listdir('Faces'):
            image_path = os.path.join('Faces', image_name)
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(os.path.splitext(image_name)[0])
        print(f"Encoding faces took {time() - start_time} seconds")

    def send_message_to_arduino(self, message):
        self.arduino.write(message.encode())
        print(f"Message sent to Arduino: {message}")

    def receive_confirmation_from_arduino(self):
        while True:
            if self.arduino.in_waiting > 0:
                confirmation = self.arduino.readline().decode().strip()
                if confirmation == "ACK":
                    self.confirmation_received = True
                    print("Confirmation received from Arduino")
                    break

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        process_frame_count = 0
        while True:
            ret, frame = video_capture.read()

            if not ret:
                continue

            process_frame_count += 1
            if process_frame_count % 3 != 0:  # Skip some frames
                continue

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = '???'

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                    # if float(confidence[:-1]) > 85 and not self.confirmation_received:
                    #     self.send_message_to_arduino("FACE_DETECTED")
                    #     self.receive_confirmation_from_arduino()

                self.face_names.append(f'{name} ({confidence})')
                if float(confidence[:-1]) > 90:
                    self.send_message_to_arduino("FACE_DETECTED")
                    self.receive_confirmation_from_arduino()

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        self.arduino.close()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
