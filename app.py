'''WEBSITE BUILD USING FLASK FRAMEWORK'''
from flask import Flask, render_template, request
from Website.Codes import code as exe
from werkzeug.utils import secure_filename
import os
from Website.Codes import db

folder = '/Users/sadhvik/Desktop/FinalProject/Website/static/images/Unprocessed'

app = Flask(__name__)

#ROUTING WEBSITES
@app.route('/',methods=['GET','POST'])
@app.route('/homepage.html',methods=['GET','POST'])
def homepage():
    return render_template('homepage.html')

@app.route('/scanleaf.html',methods=['GET','POST'])
def scanleaf():
    if request.method == 'POST':
        image = request.files['files']
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        plant,disease,sym,tre = exe.compute(filename)
        print('File(s) successfully saved')
        x = render_template('result.html', cimage=filename)
        x += "<center><section class='wrapper' style='overflow: scroll;'>"
        x += "<div class='heade' style='overflow: scroll;'><h2> " + str(plant) + " " + str(disease) + "</h2></div><center><br>"
        x += "<u><div class='sym' style='overflow: scroll;'><center><h3>Symptoms</h3></center></u>" + str(sym) + "</div>"
        x += "<u><div class='tre' style='overflow: scroll;'><center><h3>Treatment</h3></center></u>" + str(tre) + "</div></section>"
        return x
    else:
        return render_template('scanleaf.html')

@app.route('/database.html',methods=['GET','POST'])
def database():
    if request.method == 'POST':
        plant = request.form.get('pn')
        disease = request.form.get('dn')
        sym,tre = db.get_sym_tre(plant,disease)
        x= render_template('database.html')
        x+= "<body><section class='wrapper' style='overflow: scroll;'>"
        x+= "<div class='heade' style='overflow: scroll;'><h2> "+str(plant)+" "+str(disease)+"</h2></div><br>"
        x+= "<u><div class='sym' style='overflow: scroll;'><center><h3>Symptoms</h3></center></u>"+str(sym)+"</div>"
        x+= "<u><div class='tre' style='overflow: scroll;'><center><h3>Treatment</h3></center></u>"+str(tre)+"</div></section></body>"
        return x
    else:
        return render_template('database.html')


#RUNNING WEBSITE
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = folder
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config["CACHE_TYPE"] = "null"
    app.run(host='0.0.0.0', port=5115)
