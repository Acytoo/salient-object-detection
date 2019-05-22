#include "process_single_page.h"
#include "ui_process_single_page.h"

#include <thread>
#include <iostream>

#include <qmessagebox.h>
#include <qfiledialog.h>

#include <saliency/saliency_region_contrast.h>

using std::string;

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

  string res_salient, res_salient_bi, res_salient_cut;
  std::thread img_process_thread(regioncontrast::RegionContrast::ProcessSingleImg, std::ref(image_path_), std::ref(res_salient),
                                 std::ref(res_salient_bi), std::ref(res_salient_cut));
  img_process_thread.join();

  QPixmap img_salient(res_salient.c_str());
  QPixmap img_salient_bi(res_salient_bi.c_str());
  QPixmap img_salient_cut(res_salient_cut.c_str());
  ui->label_img_salient->setPixmap(img_salient.scaled(200, 200, Qt::KeepAspectRatio));
  ui->label_img_salient_bi->setPixmap(img_salient_bi.scaled(200, 200, Qt::KeepAspectRatio));
  ui->label_img_salient_cut->setPixmap(img_salient_cut.scaled(400, 400, Qt::KeepAspectRatio));

  QString original_path = "Original image, " + QString::fromStdString(image_path_);
  QString res_salent_cut_path = "Cut, " + QString::fromStdString(res_salient_cut);

  ui->label_original_path->setText(original_path);
  ui->label_result_rcc_path->setText(res_salent_cut_path);
}
