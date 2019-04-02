#ifndef SERVER_BASIC_FILE_WIN_H_
#define SERVER_BASIC_FILE_WIN_H_

#include <iostream>
#include <vector>
#include <fstream>

#include <direct.h>
#include <io.h>


namespace ytfile {
  bool is_dir(const std::string& path);
  int mk_dir(const std::string& dir_name);
  bool file_exist(const std::string& file_path);
  int get_file_names(const std::string path,
                      std::vector<std::string>& files);
}


inline bool ytfile::is_dir(const std::string& path) {
  struct _stat file_stat;
  return (_stat(path.c_str(), &file_stat) == 0) && (file_stat.st_mode & _S_IFDIR);

}

inline int ytfile::mk_dir(const std::string& dir_name) {
  return _mkdir(dir_name.c_str());
}

inline bool ytfile::file_exist(const std::string &file_path) {
  return _access(file_path.c_str(), 0) == 0;
}

int ytfile::get_file_names(const std::string folder,
                            std::vector<std::string>& files) {
  intptr_t hFile = 0;  // for x64
  struct _finddata_t fileinfo;
  string p;
  if((hFile = _findfirst(p.assign(folder).append("\\*").c_str(), &fileinfo)) !=  -1)
    do {
      if(!(fileinfo.attrib & _A_SUBDIR)) // not a sub folder
        files.push_back(fileinfo.name);

    } while (_findnext(hFile, &fileinfo)  == 0);

  _findclose(hFile);
  return 0;
}


#endif
