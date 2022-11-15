import os
from flask import flash, request, redirect, render_template, Flask
from werkzeug.utils import secure_filename
import subprocess

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		UPLOAD_FOLDER=app.config['UPLOAD_FOLDER']
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		if os.path.exists(file_path):
			os.remove(file_path)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		output=subprocess.check_output(['python','prediction.py','-image_file_name',UPLOAD_FOLDER+filename])
		output=output.decode("utf-8").split('/n')
		start = output[0].find("start ") + len("start ")
		end = output[0].find(" end")
		substring = output[0][start:end]
		check = substring.split("_")
		img_name = substring+".jpg"
		image_path = UPLOAD_FOLDER + img_name
		return render_template('upload.html', filename=file_path, output=substring, img=image_path, name=check[0].capitalize(), emotion=check[1].capitalize())
	else:
		flash('Allowed audio types are -> wav')
		return redirect(request.url)

if __name__ == "__main__":
    app.run(host='localhost', port=5500)
