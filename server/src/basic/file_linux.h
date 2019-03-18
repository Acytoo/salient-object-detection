#ifndef SERVER_BASIC_FILE_LINUX_H_
#define SERVER_BASIC_FILE_LINUX_H_

#include <iostream>
#include <vector>

#include <stdarg.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

namespace ytfile {
  bool is_dir(const std::string& path);
  int mk_dir(const std::string& dir_name);
  bool file_exist(const std::string& file_path);
  void get_file_names(const std::string folder_path,
                      std::vector<std::string>& file_names);
}


inline bool ytfile::is_dir(const std::string& path) {
  struct stat file_stat;
  return (stat(path.c_str(), &file_stat) == 0) && S_ISDIR(file_stat.st_mode);
}

inline int ytfile::mk_dir(const std::string& dir_name) {
  return mkdir(dir_name.c_str(), 0755);
}

inline bool ytfile::file_exist(const std::string &file_path) {
  return access(file_path.c_str(), F_OK) == 0;
}

void ytfile::get_file_names(const std::string folder_path,
                            std::vector<std::string>& file_names) {
  struct dirent *ptr;
  DIR *dir;
  dir = opendir(folder_path.c_str());
  while((ptr=readdir(dir))!=NULL) {
    if(ptr->d_name[0] == '.') //跳过'.'和'..'两个目录
      continue;
    file_names.push_back(ptr->d_name);
  }
}



#endif
