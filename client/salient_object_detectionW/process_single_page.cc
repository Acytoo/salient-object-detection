#include "process_single_page.h"
#include "ui_process_single_page.h"

#include <thread>
#include <iostream>

#include <qmessagebox.h>
#include <qfiledialog.h>

#include <saliency/saliency_cut.h>

process_single_page::process_single_page(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::process_single_page)
{
  std::cout << "constructor: single-page" << std::endl;
  ui->setupUi(this);
  move(150, 150);
}

process_single_page::~process_single_page()
{

  std::cout << "destructor: single-page" << std::endl;
  delete ui;
}

void process_single_page::on_button_back_clicked()
{
  emit go_back_signal();
  this->close();
  this->setAttribute(Qt::WA_DeleteOnClose,1);
}

void process_single_page::on_button_browse_clicked()
{
  file_path_ = QFileDialog::getOpenFileName(
                                            this,
                                            "Choose an image to process",
                                            QString::null,
                                            QString::null);
  if (file_path_.isEmpty()) {
    QMessageBox::about(this, "Error", "Please choose an image!");
    return;
  }
  std::string file_path =  file_path_.toUtf8().constData();
  if(!(file_path.substr(file_path.length()-4,4) == ".jpg" || file_path.substr(file_path.length()-5,5) == ".jpeg"
       || file_path.substr(file_path.length()-4,4) == ".png" || file_path.substr(file_path.length()-4,4) == ".bmp"
       )) {
    QMessageBox::about(this, "Error", "Only JPG, PNG or BMP files are supported!");
    return;
  }
  image_path_ = file_path;
  QPixmap img_ori(image_path_.c_str());
  ui->label_img_ori->setPixmap(img_ori.scaled(400,400,Qt::KeepAspectRatio));
  ui->edit_file_path->setText(file_path_);

}

void process_single_page::on_button_process_clicked()
{
  if (ui->edit_file_path->text().isEmpty()) {
    QMessageBox::about(this, "Error", "Please choose an image!");
    return;
  }
  std::string result_path_rc, result_path_rcc;
  std::thread img_process_thread(saliencycut::SaliencyCut::ProcessSingleImg, std::ref(image_path_), std::ref(result_path_rc), std::ref(result_path_rcc));
  img_process_thread.join();

  QPixmap img_rc(result_path_rc.c_str());
  ui->label_img_rc->setPixmap(img_rc.scaled(400,400,Qt::KeepAspectRatio));
  QPixmap img_rcc(result_path_rcc.c_str());
  ui->label_img_rcc->setPixmap(img_rcc.scaled(400,400,Qt::KeepAspectRatio));

  QString original_path = "Original image, " + QString::fromStdString(image_path_);
  QString result_rc_path = "RC, " + QString::fromStdString(result_path_rc);
  QString result_rcc_path = "RCC, " + QString::fromStdString(result_path_rcc);
  ui->label_original_path->setText(original_path);
  ui->label_result_rc_path->setText(result_rc_path);
  ui->label_result_rcc_path->setText(result_rcc_path);
}
