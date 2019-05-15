#ifndef BASIC_IMAGE_OPERATIONS_H
#define BASIC_IMAGE_OPERATIONS_H

#include <iostream>
#include <QMainWindow>

namespace Ui {
  class basic_image_operations;
}

class basic_image_operations : public QMainWindow
{
  Q_OBJECT

public:
  explicit basic_image_operations(QWidget *parent = nullptr);
  ~basic_image_operations();

signals:
  void go_back_signal();


private slots:
  void on_button_back_clicked();

  void on_button_browse_clicked();

  void on_button_segment_clicked();

  // void on_button_resize_clicked();

  void on_button_bgr2lab_clicked();

  void on_button_gaussian_blur_clicked();

private:
  Ui::basic_image_operations *ui;
  QString file_path_;
  std::string image_path_;
};

#endif // BASIC_IMAGE_OPERATIONS_H
