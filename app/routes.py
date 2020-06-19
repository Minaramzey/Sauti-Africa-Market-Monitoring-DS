from flask import Flask, render_template, request, Markup
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt 
import io
import base64
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from app.modules.code.db_connect import dbConnect
from app import flask_app  

db_c = dbConnect()
@app.route("/") 
def home(): 
    return "<h1>Sauti East African Market Prediction. </h1>"

@app.route("/qc/<tablename>", methods=('POST', 'GET'))
def qc_tables(tablename='qc_retail'):
    df = db_c.read_analytical_db(tablename)
    df = df.sort_values('DQI', ascending=False)
    tables = [df.to_html(classes='data')]
    return render_template("table.html", tables=tables, titles=df.columns.values, title=tablename)

@app.route("/forecast/plot")
def forecast_plot():
    # generate plot
    y = [1,2,3,4,5]
    x = [0,2,1,3,4]
    plt.plot(x,y)
    plt.savefig('static/test.png')
    
    # # render plot in html
    # img = io.BytesIO()
    # plt.savefig(img, format='png')
    # img.seek(0)
    # plot_url = base64.b64encode(img.getvalue()).decode()
    # model_plot = Markup('<img src="data:image/png;base64,{}" width: 360px; height: 288px>'.format(plot_url))
    return '<img src="/static/test.png">'
    
    #render_template('plot.html')#, model_plot = model_plot)
    # return send_file(img,
    # attachment_filename=filename,
    # mimetype='image/png')

    # f = plt.figure()
    # canvas = FigureCanvas(f)
    # canvas.print_png(img)
   
    # image_url = "data:image/png;base64,"+plot_data
    # 


