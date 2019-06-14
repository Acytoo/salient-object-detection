#include "process_multiple_page.h"
#include "ui_process_multiple_page.h"

#include <thread>
#include <iostream>

#include <qmessagebox.h>
#include <qfiledialog.h>

#include <saliency/saliency_region_contrast.h>

using namespace std;

process_multiple_page::process_multiple_page(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::process_multiple_page)
{
  std::cout << "constructor: multiple-page" << std::endl;
  ui->setupUi(this);
  move(120, 120);
}

process_multiple_page::~process_multiple_page()
{
  std::cout << "destructor: multiple-page" << std::endl;
  delete ui;
}

void process_multiple_page::on_button_back_clicked()
{
  emit go_back_signal();
  this->close();
  this->setAttribute(Qt::WA_DeleteOnClose,1);
}


void process_multiple_page::on_button_browse_clicked()
{
  ui->label_finish->setText("Press 'PROCESS' button to start\nPlease be patient");
  QString path = QFileDialog::getExistingDirectory(this, tr("Choose directory"), ".", QFileDialog::ReadOnly);
  ui->edit_root_path->setText(path);
}

void process_multiple_page::on_button_process_clicked()
{
  if (ui->edit_root_path->text().isEmpty()) {
    QMessageBox::about(this, "Error", "Please choose a directory first!");
    return;
  }
  int amount= 0, time_cost = 0;
  double average_precision = 0.0, average_recall = 0.0, average_f = 0.0, cut_threshold = 1.55;
  std::string root_dir_path = ui->edit_root_path->text().toUtf8().constData();
  bool benchmark = ui->radio_benchmark->isChecked();
  std::thread img_process_thread(regioncontrast::RegionContrast::ProcessImages,
                                std::ref(root_dir_path), std::ref(amount),
                                std::ref(time_cost), std::ref(benchmark),
                                std::ref(average_precision), std::ref(average_recall),
                                 std::ref(average_f), std::ref(cut_threshold));
  img_process_thread.join();
  string result_summary = "Processed " + to_string(amount) + " images in " + to_string(time_cost) + " seconds!";
  ui->label_finish->setText(QString::fromStdString("Processed " + to_string(amount) + " images in " + to_string(time_cost) + " seconds!"));
  if (benchmark)
    ui->label_precision->setText(QString::fromStdString("Average precision for the " + to_string(amount) + " images is " + to_string(average_precision)
                                                        + ", average recall is " + to_string(average_recall)
                                                        + ", average F1 score is " + to_string(average_f)));
}
