from PyQt5 import QtCore, QtGui, QtWidgets
from Extraction_Package.Extract_Ui import Ui_Form,My_Progress_Form,My_Error_Form
from Extraction_Package.Ratioxlsdataconvert import MSI_xls_data, MSI_xls_data_New
import os,sys,cv2
import numpy as np

class My_Extraction_Form(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_1.clicked.connect(self.xls_open)
        self.pushButton_2.clicked.connect(self.np_save)
        self.pushButton_4.clicked.connect(self.extract)
        self.label_2.setText("")

    def xls_open(self):
        path = os.getcwd()
        file_name,_= QtWidgets.QFileDialog.getOpenFileName(self,u'Choose File',path,'Excel files (*.xls *.xlsx)')
        if file_name.split('.')[-1].lower() == 'xls':
            self.xls_path = file_name
        self.lineEdit_1.setText(file_name)

    def np_save(self):
        fname, ftype = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', './', "npy files (*.npy)")
        if fname:
            self.lineEdit_3.setText(fname)

    def extract(self):
        try:
            self.xls_path = self.lineEdit_1.text()
            self.npy_save_path = self.lineEdit_3.text()
            self.mbt = extract_thread(self.xls_path,self.npy_save_path,224,224)
            self.progressBar = My_Progress_Form()
            self.progressBar.progressBar.setValue(0)
            self.progressBar.pushButton.setVisible(True)
            self.progressBar.pushButton.setText('Cancel')
            self.progressBar.pushButton.clicked.connect(self.thread_terminate)
            self.progressBar.show()
            self.mbt.trigger.connect(self.progress_update)
            self.mbt.start()
        except Exception as e :
            m='Running error, info: '+str(e)
            self.error(m)

    def error(self,m):
        self.eW=My_Error_Form()
        self.eW.label.setText(m)
        self.eW.show()

    def progress_update(self,val,stry):
        if val!=-1:
            self.progressBar.progressBar.setValue(val)
            if val==100:
                self.progressBar.label.setText('Finished!')
                self.clo = close_widget_thread(1)
                self.clo.start()
                self.clo.trigger.connect(self.close_progressbar)
        else:
            self.progressBar.label.setText('Running error, info: '+stry)
            self.progressBar.progressBar.setValue(0)
            self.progressBar.pushButton.setVisible(True)

    def close_progressbar(self,val):
        if val==100:
            self.progressBar.close()

    def thread_terminate(self):
        self.mbt.terminate()

class extract_thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal(int,str)

    def __init__(self,a,b,c,d):
        super().__init__()
        self.xls_path = a
        self.npy_save_path = b
        self.size_x , self.size_y = c,d

    def run(self):
        try:
            self.trigger.emit(5,"")
            final_grid_all = []
            percent_all = [(0,0,0,0),(0.2,-0.1,-0.1,0.1),(0.2,0.2,0.2,0.2),(-0.1,0.2,0.2,-0.1),
                           (0.05,-0.05,0.1,-0.1),(0.2,-0.05,-0.05,0.15),(0.1,0.1,0.1,0.1)]

            for key in range(0,7):
                final_grid = []
                v1 = int(10*(key+1))
                self.trigger.emit(v1,'')
                for index in range(0,15):
                    v2 = int((index+1)/15*9)+v1
                    self.trigger.emit(v2,'')
                    tmp_grid = MSI_xls_data_New(self.xls_path,index,self.size_x,self.size_y,percent=percent_all[key])
                    if tmp_grid.max() != 0:
                        tmp_grid = tmp_grid / tmp_grid.max()
                    final_grid.append(tmp_grid)
                final_grid = cv2.merge(final_grid)
                final_grid_all.append(final_grid)

            tmp_grid = final_grid_all[0]
            final_grid_all.append(np.fliplr(tmp_grid))
            final_grid_all.append(np.flipud(tmp_grid))
            tmp = np.rot90(tmp_grid)
            final_grid_all.append(tmp)
            for key in range(0,2):
                tmp = np.rot90(tmp)
                final_grid_all.append(tmp)

            print(self.npy_save_path)

            dir_path = self.npy_save_path.split('.')[0]

            for key in range(0,len(final_grid_all)):
                v1 = int((key / len(final_grid_all))*15)+80
                self.trigger.emit(v1,'')
                save_path = dir_path+'_'+str(key+1)+'.npy'
                np.save(save_path,final_grid_all[key])
            print('Successfully Saved!')
            self.trigger.emit(100,'')
        except Exception as e:
            m = 'Running error, info: ' + str(e)
            self.trigger.emit(-1, m)

class close_widget_thread(QtCore.QThread):
    trigger = QtCore.pyqtSignal(int)

    def __init__(self,seconds):
        super().__init__()
        self.second = seconds

    def run(self):
        self.sleep(self.second)
        self.trigger.emit(100)

if __name__ == "__main__" :
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = My_Extraction_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())