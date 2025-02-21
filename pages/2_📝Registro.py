import streamlit as st
from app import face_rec
import cv2
import numpy as np

st.subheader('📝 - Registrar Usuario')

# Init Registration Form
registration_form = face_rec.RegistrationForm()

# Form inputs
person_name = st.text_input(label='Nombre', placeholder='Nombre y Apellido')
role = st.selectbox(label='Seleccionar Cargo', 
                   options=('Docente', 'Administrativo', 'Servicios'))

st.divider()
st.markdown('Iniciar cámara y registrar al menos 200 ejemplos')

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create placeholder for video
frame_placeholder = st.empty()

stop_button = st.button("Detener")

while not stop_button:
    ret, frame = cam.read()
    if ret:
        # Get embeddings
        reg_img, embedding = registration_form.get_embeddings(frame)
        
        if embedding is not None:
            with open('face_embedding.txt', mode='ab') as f:
                np.savetxt(f, embedding)

        # Convert BGR to RGB for Streamlit
        reg_img = cv2.cvtColor(reg_img, cv2.COLOR_BGR2RGB)
        # Display the frame
        frame_placeholder.image(reg_img)

# Release camera when stopped
cam.release()

# Save data section
st.markdown('Presionar Guardar, para almacenar los datos.')

if st.button('Guardar'):
    return_val = registration_form.save_data_in_redis_db(person_name, role)
    if return_val == True:
        st.success(f'{person_name} registrado exitosamente')
    elif return_val == 'name_false':
        st.error('Ingrese el nombre: no dejar vacio o con espacios.')
    elif return_val == 'file_false':
        st.error('Por favor refresque la página y ejecute nuevamente.')

st.divider()

with st.expander('Eliminar usuario:'):
    
    person_name_del = st.text_input(label='Nombre Eliminar', placeholder='Nombre y Apellido')
    role_del = st.selectbox(label='Seleccionar Cargo Eliminar', options=('Docente', 'Administrativo', 'Servicios'))
    st.write(person_name)

    if st.button('Eliminar'):
        return_eli = registration_form.delete_user_from_redis(person_name, role)
        if return_eli == True:
            st.success(f'{person_name} eliminado exitosamente')
        elif return_eli == 'name_false':
            st.error('Ingrese el nombre: no dejar vacio o con espacios.')

        elif return_eli == 'file_false':
            st.error('Por favor refresque la página y ejecute nuevamente.')