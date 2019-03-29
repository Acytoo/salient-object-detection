#ifndef PROCESS_MULTIPLE_PAGE_H
#define PROCESS_MULTIPLE_PAGE_H

#include <QMainWindow>

namespace Ui {
  class process_multiple_page;
}

class process_multiple_page : public QMainWindow
{
  Q_OBJECT

public:
  explicit process_multiple_page(QWidget *parent = nullptr);
  ~process_multiple_page();

signals:
  void go_back_signal();

private slots:
  void on_button_back_clicked();


  void on_button_browse_clicked();

  void on_button_process_clicked();

private:
  Ui::process_multiple_page *ui;
};

#endif // PROCESS_MULTIPLE_PAGE_H
