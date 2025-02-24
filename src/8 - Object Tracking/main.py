import cv2
import mediapipe as mp
import numpy as np

# Initialiser MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

def detect_and_track_person(frame):
    # Convertir l'image en RGB pour MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détecter la personne
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # Obtenir les coordonnées du rectangle englobant
        landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] 
                            for lm in results.pose_landmarks.landmark])
        
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        # Convertir en entiers
        bbox = [
            int(x_min), int(y_min),
            int(x_max - x_min), int(y_max - y_min)
        ]
        
        # Dessiner le squelette
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
        )
        
        # Dessiner le rectangle englobant
        cv2.rectangle(frame, 
                     (bbox[0], bbox[1]), 
                     (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                     (0, 255, 0), 2)
        
        # Ajouter le texte
        cv2.putText(frame, "Personne detectee", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return True, bbox
    
    return False, None

def main():
    # Initialiser la capture vidéo
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    while True:
        # Lire une frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Détecter et suivre la personne
        success, bbox = detect_and_track_person(frame)
        
        if not success:
            cv2.putText(frame, "Aucune personne detectee", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Afficher la frame
        cv2.imshow("Suivi de personne", frame)
        
        # Quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()