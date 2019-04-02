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
  int get_file_names(const std::string folder_path,
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

// return value: 0: no error; -1: not a folder; -2: others
int ytfile::get_file_names(const std::string folder_path,
                            std::vector<std::string>& file_names) {
  struct stat s;
  const char* dir_name = folder_path.c_str();
  lstat(dir_name, &s);
  if(!S_ISDIR(s.st_mode))
    return -1;
  struct dirent *ptr;
  DIR *dir;
  dir = opendir(dir_name);
  while((ptr=readdir(dir))!=NULL) {
    if(ptr->d_type == 8)
      file_names.push_back(ptr->d_name);
  }
  return 0;
}
/*有些情况下，我们只要输出文件而不需要文件夹（目录），这时可以通过dirent结构体中的d_type进行过滤。d_type表示类型，4表示目录，8表示普通文件，0表示未知设备。*/


#endif
