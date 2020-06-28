import sqlite3 as db


def send_list():
    #sending list of diseases to code.py for naming output classes in neural network
    con = db.connect('LeafDiseaseDatabase.db')
    cur = con.cursor()
    sqlcmd = "SELECT Plant,Disease from MAIN"
    cur.execute(sqlcmd)
    data = cur.fetchall()
    info,plant,disease = [],[],[]
    for i,j in data:
        plant.append(i)
        disease.append(j)
        info.append(j+" ("+i+")")
    con.close()
    return plant, disease, info

def get_sym_tre(plant,disease):
    #getting information of diseases once detection
    con = db.connect('LeafDiseaseDatabase.db')
    cur = con.cursor()
    sqlcmd = "SELECT Symptoms,Treatment from MAIN where Plant='"+plant+"' AND Disease='"+disease+"'"
    cur.execute(sqlcmd)
    data = cur.fetchall()
    sym,tre = data[0][0],data[0][1]
    con.close()
    return sym, tre