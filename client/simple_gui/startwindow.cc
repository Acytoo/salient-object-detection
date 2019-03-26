#include "startwindow.h"
#include "ui_startwindow.h"
#include "process_single_page.h"
#include "process_multiple_page.h"

StartWindow::StartWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::StartWindow)
{
  ui->setupUi(this);
  move(90, 90);
}

StartWindow::~StartWindow()
{
  delete ui;
}

void StartWindow::on_button_single_clicked()
{
  this->hide();
  process_single_page *single_page = new process_single_page(this);
  connect(single_page, SIGNAL(go_back_signal()), this, SLOT(reshow()));
  single_page->show();
}

void StartWindow::reshow() {
  this->show();
}

void StartWindow::on_button_multiple_clicked()
{
  this->hide();
  //process_single_page *single_page = new process_single_page(this);
  process_multiple_page *multiple_page = new process_multiple_page(this);
  connect(multiple_page, SIGNAL(go_back_signal()), this, SLOT(reshow()));
  multiple_page->show();
}
