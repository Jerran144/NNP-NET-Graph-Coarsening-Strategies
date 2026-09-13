#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace NNPNet {
class Utils {
public:
  static std::string getFirstTokenInString(const std::string &s,
                                           const char del = ' ') {
    int i = 0;
    while (i < s.size() && s[i] != del)
      i++;
    return s.substr(0, i);
  }

  static std::vector<std::string> split(const std::string &s,
                                        const char del = ' ') {
    std::vector<std::string> output;
    int start = 0;
    for (int i = 0; i < s.size(); i++) {
      if (s[i] == del) {
        output.push_back(s.substr(start, i - start));
        start = i + 1;
      }
    }
    output.push_back(s.substr(start));

    return output;
  }

private:
  Utils();
};

typedef std::chrono::high_resolution_clock Clock;

class StatsCollector {
public:
  inline static std::map<std::string, double> times;
  static void clear() { times.clear(); }
};

#define TIME(toTime, name)                                                     \
  {                                                                            \
    Clock __c;                                                                 \
    auto __start = __c.now();                                                  \
    toTime;                                                                    \
    double __dur = (__c.now() - __start).count() / 1000000000.0;               \
    std::cout << "Time " name ": " << __dur << "\n";                           \
    NNPNet::StatsCollector::times[name] = __dur;                               \
  }

}; // namespace NNPNet