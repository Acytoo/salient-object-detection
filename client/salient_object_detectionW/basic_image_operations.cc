#include "basic_image_operations.h"
#include "ui_basic_image_operations.h"

#include <iostream>
#include <thread>

#include <qmessagebox.h>
#include <qfiledialog.h>

#include <basic/basic_functions_demo.h>

basic_image_operations::basic_image_operations(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::basic_image_operations)
{
  std::cout << "constructor: basic-operations-page" << std::endl;
  ui->setupUi(this);
  move(150, 150);
}

basic_image_operations::~basic_image_operations()
{
  std::cout << "destructor: basic-operations-page" << std::endl;
  delete ui;
}

void basic_image_operations::on_button_back_clicked()
{
  emit go_back_signal();
  this->close();
  this->setAttribute(Qt::WA_DeleteOnClose,1);
}

void basic_image_operations::on_button_browse_clicked()
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
}

void basic_image_operations::on_button_segment_clicked()
{
  if (file_path_.isEmpty()) {
    QMessageBox::about(this, "Error", "Please choose an image!");
    return;
  }
  std::string seg_path = "";
  std::thread img_process_thread(basic_functions_demo::segment_demo, std::ref(image_path_), std::ref(seg_path));
  img_process_thread.join();
  QPixmap seg_img(seg_path.c_str());
  ui->label_img_processed->setPixmap(seg_img.scaled(400,400,Qt::KeepAspectRatio));
  ui->label_function_name->setText("Segment Image");
}

// void basic_image_operations::on_button_resize_clicked()
// {
//   if (file_path_.isEmpty()) {
//     QMessageBox::about(this, "Error", "Please choose an image!");
//     return;
//   }
// }

void basic_image_operations::on_button_bgr2lab_clicked()
{
  if (file_path_.isEmpty()) {
    QMessageBox::about(this, "Error", "Please choose an image!");
    return;
  }
  std::string lab_path = "";
  std::thread img_process_thread(basic_functions_demo::bgr2lab_demo, std::ref(image_path_), std::ref(lab_path));
  img_process_thread.join();
  QPixmap seg_img(lab_path.c_str());
  ui->label_img_processed->setPixmap(seg_img.scaled(400,400,Qt::KeepAspectRatio));
  ui->label_function_name->setText("BGR2Lab");
}
