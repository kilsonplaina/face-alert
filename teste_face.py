import cv2
import face_recognition

print("🔎 Iniciando FaceAlert - Teste de Câmera e Rosto")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Não foi possível abrir a câmera.")
    print("👉 Troca para 1 ou 2: cap = cv2.VideoCapture(1)")
    raise SystemExit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Falha ao capturar imagem")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb, model="hog")

    for (top, right, bottom, left) in faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Rostos detectados: {len(faces)}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FaceAlert - TESTE (ESC para sair)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Teste finalizado")
