from flask import Flask, request, render_template
from flask import redirect, url_for, send_file
import subprocess
from datetime import datetime
import os
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    now = datetime.now()
    date_str = now.strftime("%Y/%m/%d")
    os.makedirs(date_str, exist_ok=True)
    dir = './'+date_str+'/cropped_images'
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
    else:
        os.makedirs(dir)
    
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_faces():
    # Recebe a imagem/vídeo enviada pelo cliente
    file = request.files['file']    
    
    # Salva o arquivo em um diretório com a data atual
    now = datetime.now()
    date_str = now.strftime("%Y/%m/%d")
    os.makedirs(date_str, exist_ok=True)
    filepath = os.path.join(date_str, file.filename)
    file.save(filepath)
    dir = './'+date_str+'/cropped_images'
   
    # Chama o programa de detecção de faces com o caminho do arquivo salvo
    # Substitua 'programa_de_detecao_de_faces.py' pelo nome do seu programa
    # que realiza a detecção de faces
    # O resultado da detecção de faces pode ser retornado como uma resposta JSON
    # ou como um arquivo de imagem/vídeo modificado
    result = subprocess.run(['python', '/app/Detection/detect.py', filepath], capture_output=True)
    
    return redirect(url_for('images_det'))

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    # Recebe a imagem/vídeo enviada pelo cliente
    file = request.files['file']

    now = datetime.now()
    date_str = now.strftime("%Y/%m/%d")
    os.makedirs(date_str, exist_ok=True)
    filepath = os.path.join(date_str, file.filename)  
    file.save(filepath)
    dir = './'+date_str+'/cropped_images'
    
    # Chama o programa de reconhecimento de faces
    # Substitua 'recognize.py' pelo nome do seu programa
    # que realiza o reconhecimento de faces
    # O resultado do reconhecimento de faces pode ser retornado como uma resposta JSON
    # ou como um arquivo de imagem/vídeo modificado
    result = subprocess.run(['python', '/app/Recognition/arcface_torch/recognize.py', filepath], capture_output=True)
    
    return redirect(url_for('images_rec'))

@app.route('/images_det', methods=['GET'])
def images_det():
    now = datetime.now()
    date_str = now.strftime("%Y/%m/%d")
    os.makedirs(date_str, exist_ok=True)
    path = os.path.join(os.getcwd(), date_str, 'cropped_images')
    imagens = [os.path.join(date_str, 'cropped_images', f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return render_template('faces_detected.html',imagens=imagens)
   
@app.route('/image/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_file(filename, mimetype='image/jpeg')

@app.route('/images_rec', methods=['GET'])
def images_rec():
    now = datetime.now()
    date_str = now.strftime("%Y/%m/%d")
    os.makedirs(date_str, exist_ok=True)
    path = os.path.join(os.getcwd(), date_str, 'cropped_images')
    imagens = [os.path.join(date_str, 'cropped_images', f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    classes = []
    for imagem in imagens:
        nome_imagem = os.path.basename(imagem)
        classe = nome_imagem.split('.')
        classes.append(classe[0])
    
    zipped = zip(classes, imagens)

    return render_template('faces_recognized.html', zipped=zipped)

if __name__ == '__main__':
    app.run()

