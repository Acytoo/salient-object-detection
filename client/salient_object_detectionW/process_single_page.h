#ifndef PROCESS_SINGLE_PAGE_H
#define PROCESS_SINGLE_PAGE_H

#include <QMainWindow>
#include <iostream>

namespace Ui {
  class process_single_page;
}

class process_single_page : public QMainWindow
{
  Q_OBJECT

public:
  explicit process_single_page(QWidget *parent = nullptr);
  ~process_single_page();

    std::string res_salient, res_salient_bi, res_salient_cut;
signals:
  void go_back_signal();

private slots:
  void on_button_back_clicked();

  void on_button_browse_clicked();

  void on_button_process_clicked();

  void ShowContextMenu(const QPoint &pos);
  void cutsave_as();
  void bisave_as();
  void rcsave_as();

private:
  Ui::process_single_page *ui;
  QString file_path_;
  std::string image_path_;
};

#endif // PROCESS_SINGLE_PAGE_H
