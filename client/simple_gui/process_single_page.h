#ifndef PROCESS_SINGLE_PAGE_H
#define PROCESS_SINGLE_PAGE_H

#include <QMainWindow>

namespace Ui {
  class process_single_page;
}

class process_single_page : public QMainWindow
{
  Q_OBJECT

public:
  explicit process_single_page(QWidget *parent = nullptr);
  ~process_single_page();

signals:
  void go_back_signal();

private slots:
  void on_button_back_clicked();

  void on_button_browse_clicked();

  void on_button_process_clicked();

private:
  Ui::process_single_page *ui;
};

#endif // PROCESS_SINGLE_PAGE_H
