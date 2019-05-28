#include "process_single_page.h"
#include "ui_process_single_page.h"

#include <thread>
#include <iostream>

#include <qmessagebox.h>
#include <qfiledialog.h>
#include <QMenu>

#include <saliency/saliency_region_contrast.h>

using std::string;

process_single_page::process_single_page(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::process_single_page)
{
  std::cout << "constructor: single-page" << std::endl;
  ui->setupUi(this);
  move(150, 150);

  this->setContextMenuPolicy(Qt::CustomContextMenu);

  connect(this, SIGNAL(customContextMenuRequested(const QPoint &)),
          this, SLOT(ShowContextMenu(const QPoint &)));
}

void process_single_page::ShowContextMenu(const QPoint &pos)
{
  QRect rect_cut(830, 110, 400, 400), rect_rc(510, 80, 200, 200), rect_bi(510, 350, 200, 200), rect_ori(10, 110, 400, 400);
  //string res_salient, res_salient_bi, res_salient_cut;
  if (rect_cut.contains(pos)) {
    QMenu contextMenu(tr("Menu for cut"), this);
    QAction action1("Cut: Save as", this);
    connect(&action1, SIGNAL(triggered()), this, SLOT(cutsave_as()));
    contextMenu.addAction(&action1);
    contextMenu.exec(mapToGlobal(pos));
  } else if (rect_rc.contains(pos)) {
    QMenu contextMenu(tr("Menu for rc"), this);
    QAction action1("Salient: Save as", this);
    connect(&action1, SIGNAL(triggered()), this, SLOT(rcsave_as()));
    contextMenu.addAction(&action1);
    contextMenu.exec(mapToGlobal(pos));
  } else if (rect_bi.contains(pos)) {
    QMenu contextMenu(tr("Menu for bi"), this);
    QAction action1("Bi: Save as", this);
    connect(&action1, SIGNAL(triggered()), this, SLOT(bisave_as()));
    contextMenu.addAction(&action1);
    contextMenu.exec(mapToGlobal(pos));
  }
}


void process_single_page::cutsave_as() {
  //QString FilePathName = "/home/acytoo/delete.png";
  QString FilePathName = QString::fromStdString(res_salient_cut);
  QFileDialog m_QFileDialog;
  QString setFilter = "image(*.png);;";
  QString selectFilter,dirString;

  if( FilePathName.isEmpty() ) return;
  else dirString = QFileInfo(FilePathName).fileName();
  QString saveFileName = m_QFileDialog.getSaveFileName(this,"保存文件",dirString,setFilter,&selectFilter,
                                                       QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

  if( saveFileName.isEmpty())
    return;

  QFile file(saveFileName);
  bool copy_error =  file.copy( FilePathName,saveFileName );

  Q_UNUSED(copy_error);

}
void process_single_page::rcsave_as() {
  //QString FilePathName = "/home/acytoo/delete.png";
  QString FilePathName = QString::fromStdString(res_salient);
  QFileDialog m_QFileDialog;
  QString setFilter = "image(*.png);;";
  QString selectFilter,dirString;

  if( FilePathName.isEmpty() ) return;
  else dirString = QFileInfo(FilePathName).fileName();
  QString saveFileName = m_QFileDialog.getSaveFileName(this,"保存文件",dirString,setFilter,&selectFilter,
                                                       QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

  if( saveFileName.isEmpty())
    return;

  QFile file(saveFileName);
  bool copy_error =  file.copy( FilePathName,saveFileName );

  Q_UNUSED(copy_error);

}
void process_single_page::bisave_as() {
  //QString FilePathName = "/home/acytoo/delete.png";
  QString FilePathName = QString::fromStdString(res_salient_bi);
  QFileDialog m_QFileDialog;
  QString setFilter = "image(*.png);;";
  QString selectFilter,dirString;

  if( FilePathName.isEmpty() ) return;
  else dirString = QFileInfo(FilePathName).fileName();
  QString saveFileName = m_QFileDialog.getSaveFileName(this,"保存文件",dirString,setFilter,&selectFilter,
                                                       QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

  if( saveFileName.isEmpty())
    return;

  QFile file(saveFileName);
  bool copy_error =  file.copy( FilePathName,saveFileName );

  Q_UNUSED(copy_error);

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
    QMessageBox::about(this, "Error", "Please choose an image first!");
    return;
  }
  std::string file_path =  file_path_.toUtf8().constData();
  if(!(file_path.substr(file_path.length()-4,4) == ".jpg" || file_path.substr(file_path.length()-5,5) == ".jpeg"
       || file_path.substr(file_path.length()-4,4) == ".png" || file_path.substr(file_path.length()-4,4) == ".bmp"
       )) {
    QMessageBox::about(this, "Error", "Only image files are supported!");
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
    QMessageBox::about(this, "Error", "Please choose an image first!");
    return;
  }


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

  ui->label_salient->setText("Salient region");
  ui->label_binaried->setText("Binaried salient region");
}

