#ifndef STARTWINDOW_H
#define STARTWINDOW_H

#include <QMainWindow>

class process_single_page;
namespace Ui {
  class StartWindow;
}

class StartWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit StartWindow(QWidget *parent = nullptr);
  ~StartWindow();
  //void reshow();

signals:
  void jump_single(int num);
  void jump_multiple(int num);

private slots:
  void on_button_single_clicked();
  void reshow();

  void on_button_multiple_clicked();

private:
  Ui::StartWindow *ui;
  //process_single_page *single_page;
};

#endif // STARTWINDOW_H
