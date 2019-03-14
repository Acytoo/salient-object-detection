#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <thread>
#include <string>

#include <qmessagebox.h>
#include <qfiledialog.h>

#include <saliency/saliency_cut.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_process_clicked()
{
    if (ui->edit_file_path->text().isEmpty()) {
           QMessageBox::about(this, "Error", "Please choose an image!");
           return;
       }
       std::string file_path = ui->edit_file_path->text().toUtf8().constData();
       if(!(file_path.substr(file_path.length()-4,4) == ".jpg" || file_path.substr(file_path.length()-5,5) == ".jpeg"
            || file_path.substr(file_path.length()-4,4) == ".png" || file_path.substr(file_path.length()-4,4) == ".JPG"
           )) {
               QMessageBox::about(this, "Error", "Only JPG or PNG files are supported!");
               return;
           }
       QPixmap img_ori(file_path.c_str());
       ui->label_img_ori->setPixmap(img_ori.scaled(400,400,Qt::KeepAspectRatio));
       std::string result_path_rc, result_path_rcc;
       std::thread img_process_thread(saliencycut::SaliencyCut::ProcessSingleImg, std::ref(file_path), std::ref(result_path_rc), std::ref(result_path_rcc));
       img_process_thread.join();

       QPixmap img_rc(result_path_rc.c_str());
       ui->label_img_rc->setPixmap(img_rc.scaled(400,400,Qt::KeepAspectRatio));
       QPixmap img_rcc(result_path_rcc.c_str());
       ui->label_img_rcc->setPixmap(img_rcc.scaled(400,400,Qt::KeepAspectRatio));

       QString original_path = "Original image at " + QString::fromStdString(file_path);
       QString result_rc_path = "RC saved at " + QString::fromStdString(result_path_rc);
       QString result_rcc_path = "RCC saved at " + QString::fromStdString(result_path_rcc);
       ui->label_original_path->setText(original_path);
       ui->label_result_rc_path->setText(result_rc_path);
       ui->label_result_rcc_path->setText(result_rcc_path);
}

void MainWindow::on_btn_browse_clicked()
{
    QString path;
    path = QFileDialog::getOpenFileName(
                   this,
                   "Choose an image to process",
                   QString::null,
                   QString::null);
    ui->edit_file_path->setText(path);
}
