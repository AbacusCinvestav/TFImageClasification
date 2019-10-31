import os
import sys
import json
import matplotlib.pyplot as plt

histograms_dir = 'graphs'
histogram_logfile = None
histogram_monitor = None

t_error = {
    'xs': [],
    'ys': []
}

v_error = {
    'xs': [],
    'ys': []
}

def init(name):
    global histograms_dir
    global histogram_logfile
    global histogram_monitor

    histograms_dir = "%s/%s" %(name, histograms_dir)

    if not os.path.isdir(histograms_dir):
        os.mkdir(histograms_dir)

    histogram_logfile = "%s/histograms.json" %(histograms_dir)
    histogram_monitor = "%s/histograms.html" %(histograms_dir)

    print "[i] Histograms log in %s" %(histogram_logfile)

    source = "%s/histograms.html" %(os.path.split(os.path.abspath(__file__))[0])
    monitor = open(source, 'r')
    destination = open(histogram_monitor, 'wb')

    data = monitor.read()
    data = data.replace('%MODEL%', name)

    destination.write(data)
    destination.close()
    monitor.close()

    print "[i] Histograms web monitor: %s" %(histogram_monitor)

def save():
    log = {
        'histograms': {
            't_error': t_error,
            'v_error': v_error
        }
    }

    file = open(histogram_logfile, 'wb')
    file.write(json.dumps(log))
    file.close()

def log(h, x, y):
    prevent_duplicates = True

    if len(h['xs']) > 1:
        if (h['xs'][-1] == x or h['ys'][-1] == y):
            prevent_duplicates = False

    if prevent_duplicates:
        h['xs'].append(x)
        h['ys'].append(y)
        save()


def save_graphs():
    global histograms_dir

    print "[i] Saving plots images..."

    plt.style.use('seaborn-whitegrid')

    plt.title("Error histograms")
    plt.xlabel("error")
    plt.ylabel("ite");

    plt.plot(v_error['xs'], v_error['ys'], label='validation')
    plt.plot(t_error['xs'], t_error['ys'], label='training')

    plt.legend()

    img = "%s/histograms_errors.png" %(histograms_dir)
    plt.savefig(img)

    print "[i] %s saved" %(img)
