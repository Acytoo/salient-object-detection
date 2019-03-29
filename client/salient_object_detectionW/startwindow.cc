#include "startwindow.h"
#include "ui_startwindow.h"
#include "process_single_page.h"
#include "process_multiple_page.h"
#include "basic_image_operations.h"

StartWindow::StartWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::StartWindow)
{
  std::cout << "constructor: start-page" << std::endl;
  ui->setupUi(this);
  move(130, 130);
}

StartWindow::~StartWindow()
{
  std::cout << "destructor: start-page" << std::endl;
  delete ui;
}

void StartWindow::on_button_single_clicked()
{
  this->hide();
  process_single_page *single_page = new process_single_page(this);      // auto delete???
  connect(single_page, SIGNAL(go_back_signal()), this, SLOT(reshow()));
  single_page->show();
}

void StartWindow::reshow() {
  this->show();
}

void StartWindow::on_button_multiple_clicked()
{
  this->hide();
  process_multiple_page *multiple_page = new process_multiple_page(this);
  connect(multiple_page, SIGNAL(go_back_signal()), this, SLOT(reshow()));
  multiple_page->show();
}

void StartWindow::on_button_dmeo_clicked()
{
  this->hide();
  basic_image_operations *basic_operations_page = new basic_image_operations(this);
  connect(basic_operations_page, SIGNAL(go_back_signal()), this, SLOT(reshow()));
  basic_operations_page->show();
}
