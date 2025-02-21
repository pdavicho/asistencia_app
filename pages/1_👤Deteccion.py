import streamlit as st
from app import face_rec
import cv2
import time

# Inicializar en session_state si aún no existe
if 'data_saved' not in st.session_state:
    st.session_state['data_saved'] = False

st.subheader('👤 - Detección')

# Retrieve the data from Redis Database
with st.spinner('Esperando BD...'):
    redis_face_db = face_rec.retrive_data(name='academy:register')

st.success('Datos cargados desde BD')

# Create a button to start the camera
start_camera = st.button("Iniciar Cámara")
if start_camera:
    # Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create placeholder for video
    frame_placeholder = st.empty()

    # Initialize time variables
    waitTime = 10  # Time in sec for data save
    autoStopTime = 12  # Time in sec for auto stop
    startTime = time.time()  # Initial time for auto stop
    setTime = time.time()
    savedTime = 0
    realtimepred = face_rec.RealTimePred()

    stop_button = st.button("Detener")

    while not stop_button:
        ret, frame = cam.read()
        if ret:
            # Check if 12 seconds have passed
            if time.time() - startTime >= autoStopTime:
                st.warning('Tiempo de detección completado (12 segundos)')
                cam.release()  # Cerramos la cámara
                st.info("Cámara cerrada automáticamente")
                frame_placeholder.empty()  # Limpiamos el placeholder
                st.stop()  # Detenemos la ejecución de Streamlit
                break

            # Process frame for face recognition
            pred_img = realtimepred.face_prediction(frame, redis_face_db,
                                                'facial_features', 
                                                ['Name', 'Role'], 
                                                thresh=0.5)
            
            # Time management
            timenow = time.time()
            difftime = timenow - setTime
            remaining_time = max(0, waitTime - int(difftime))
            
            # Add time counter to frame
            cv2.putText(pred_img,
                        f"Tiempo restante: {autoStopTime - int(timenow - startTime)}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            if difftime >= waitTime:
                realtimepred.saveLogs_redis()
                setTime = time.time()
                savedTime = time.time()
                st.success('Datos registrados en la BD correctamente.')

            # Show saving message
            if time.time() - savedTime < 3:
                cv2.putText(pred_img,
                            "Datos guardados!",
                            (int(pred_img.shape[1]/4), int(pred_img.shape[0]/2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

            # Convert BGR to RGB for Streamlit
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            # Display the frame
            frame_placeholder.image(pred_img)

    # Release camera when stopped
    cam.release()
    st.info("Cámara detenida")
else:
    st.info("Presione 'Iniciar Cámara' para comenzar la detección facial")