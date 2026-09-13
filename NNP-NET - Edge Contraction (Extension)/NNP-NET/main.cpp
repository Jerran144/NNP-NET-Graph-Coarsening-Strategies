// MasterThesis.cpp : Defines the entry point for the application.
//

#include <chrono>
#include <iostream>
#include <string>

#include <filesystem>
namespace fs = std::filesystem;

#include <cstring>

#include "Graph.h"
#include "main.h"
#include <fstream>

struct RunStats {
  std::string graphName;
  double stress = -1;
  double np = -1;
  bool fallback = false;
  bool success = true;
  std::string errorMessage = "";
};
inline RunStats currentStats;

void appendToCSV(std::string csvPath, const RunStats &stats) {
  bool fileExists = fs::exists(csvPath);
  std::ofstream file(csvPath, std::ios_base::app);
  if (!fileExists) {
    file << "Graph Name,Embedding creation time,Subgraph creation time,Ground "
            "truth creation time,Training + inference time,Time "
            "NNP-NET,Stress,Neighbourhood preservation,Fallback "
            "flag,Success,Error\n";
  }
  file << stats.graphName << ",";
  file << NNPNet::StatsCollector::times["Create Embedding"] << ",";
  double subTime = NNPNet::StatsCollector::times["Create fast subgraph"];
  if (subTime == 0)
    subTime = NNPNet::StatsCollector::times["Create subgraph"];
  file << subTime << ",";
  file << NNPNet::StatsCollector::times["creating ground truth"] << ",";
  file << NNPNet::StatsCollector::times["Train and Inference"] << ",";
  file << NNPNet::StatsCollector::times["NNP-NET"] << ",";
  file << stats.stress << ",";
  file << stats.np << ",";
  file << (stats.fallback ? "True" : "False") << ",";
  file << (stats.success ? "True" : "False") << ",";
  file << stats.errorMessage << "\n";
}

#include "LayoutMethods/NNPNET.h"
#include "LayoutMethods/PivotMDS.h"
#include "LayoutMethods/tsNET.h"

#include "NeighborhoodPreservation.h"
#include "Smoothing.h"
#include "Stress.h"
#include "Threading.h"
#include "Utils.h"

using namespace NNPNet;

#include "pybind11/embed.h"
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace py::literals;

enum Argument {
  ARG_SMOOTH,
  ARG_THETA,
  ARG_GPU,
  ARG_PERP,
  ARG_METHOD,
  ARG_EMBEDDING_SIZE,
  ARG_SUBGRAPH_SIZE,
  ARG_PMDS_PIVOTS,
  ARG_OUTPATH,
  ARG_DIMENSIONS,
  ARG_TRAINING_EPOCHS,
  ARG_BATCH_SIZE,
  ARG_USE_FLOAT,
  ARG_TIME_SERIES,
  ARG_FEATURE_WEIGHT,
  ARG_CALC_STRESS,
  ARG_CALC_FEATURE_STRESS,
  ARG_CALC_NP,
  ARG_IGNORE_WEIGHTS,

  ARG_none
};

enum Method {
  METHOD_NNPNET,
  METHOD_TSNET,
  METHOD_TSNETSTAR,
  METHOD_PMDS,
};

std::unordered_map<std::string, Argument> argMap = {
    {"-l", ARG_SMOOTH},
    {"--smoothing", ARG_SMOOTH},
    {"-t", ARG_THETA},
    {"--theta", ARG_THETA},
    {"-g", ARG_GPU},
    {"--gpu", ARG_GPU},
    {"-p", ARG_PERP},
    {"--perplexity", ARG_PERP},
    {"-m", ARG_METHOD},
    {"--method", ARG_METHOD},
    {"-e", ARG_EMBEDDING_SIZE},
    {"--embedding_size", ARG_EMBEDDING_SIZE},
    {"-s", ARG_SUBGRAPH_SIZE},
    {"--subgraph_size", ARG_SUBGRAPH_SIZE},
    {"--pmds_pivots", ARG_PMDS_PIVOTS},
    {"-o", ARG_OUTPATH},
    {"--output", ARG_OUTPATH},
    {"-d", ARG_DIMENSIONS},
    {"--dimensions", ARG_DIMENSIONS},
    {"-b", ARG_BATCH_SIZE},
    {"--batch_size", ARG_BATCH_SIZE},
    {"--training_epochs", ARG_TRAINING_EPOCHS},
    {"--use_float", ARG_USE_FLOAT},
    {"-f", ARG_USE_FLOAT},
    {"--time_series", ARG_TIME_SERIES},
    {"--feature_weight", ARG_FEATURE_WEIGHT},
    {"--stress", ARG_CALC_STRESS},
    {"--feature_stress", ARG_CALC_FEATURE_STRESS},
    {"--np", ARG_CALC_NP},
    {"--ignore_weights", ARG_IGNORE_WEIGHTS},
};

void calculateMetrics(Graph<float> &g,
                      std::unordered_map<Argument, double> &settings) {
  if (settings[ARG_CALC_FEATURE_STRESS] > 0) {
    currentStats.stress = CalcStress::CalcFeatureStress(g);
  }
  if (settings[ARG_CALC_STRESS] > 0) {
    currentStats.stress = CalcStress::Calc(g);
  }
  if (settings[ARG_CALC_NP] > 0) {
    currentStats.np = NP::np(g);
  }
}

void calculateMetrics(Graph<double> &g,
                      std::unordered_map<Argument, double> &settings) {
  if (settings[ARG_CALC_FEATURE_STRESS] > 0) {
    Graph<float> gf(g);
    currentStats.stress = CalcStress::CalcFeatureStress(gf);
  }
  if (settings[ARG_CALC_STRESS] > 0) {
    Graph<float> gf(g);
    currentStats.stress = CalcStress::Calc(gf);
  }
  if (settings[ARG_CALC_NP] > 0) {
    currentStats.np = NP::np(g);
  }
}

template <class T>
void createLayoutFor(std::string path, std::string outPath,
                     std::unordered_map<Argument, double> &settings) {
  switch ((Method)(int)settings[ARG_METHOD]) {
  case METHOD_NNPNET: {
    Graph<float> g(settings[ARG_DIMENSIONS]);
    NNPNET nnpnet;
    nnpnet.perplexity = settings[ARG_PERP];
    nnpnet.gpu = settings[ARG_GPU] > 0;
    nnpnet.useFloats = settings[ARG_USE_FLOAT] > 0;
    nnpnet.theta = settings[ARG_THETA];
    nnpnet.subgraphPoints = settings[ARG_SUBGRAPH_SIZE];
    nnpnet.pmdsPivots = settings[ARG_PMDS_PIVOTS];
    nnpnet.trainingEpochs = settings[ARG_TRAINING_EPOCHS];
    nnpnet.batchSize = settings[ARG_BATCH_SIZE];
    nnpnet.featureWeight = settings[ARG_FEATURE_WEIGHT];

    fs::path inFsPath(path);
    fs::path subPath;
    if (outPath != "") {
      subPath = fs::path(outPath).parent_path() /
                (inFsPath.stem().string() + "_subgraph_out.vna");
    } else {
      subPath = inFsPath.parent_path() /
                (inFsPath.stem().string() + "_subgraph_out.vna");
    }
    nnpnet.exportSubgraphPath = subPath.string();

    if (settings[ARG_TIME_SERIES]) {
      g.loadFromFile(path, settings[ARG_IGNORE_WEIGHTS] > 0);
      g.fillNodeNames();
      int *pivotPoints;
      { // Training
        int i = 1;
        std::string tsdPath =
            path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";

        Graph<float> _g(g);
        while (std::filesystem::exists(tsdPath)) {
          _g.loadTimeSeries(tsdPath, true);

          i++;
          tsdPath =
              path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
        }
        bool *inAll = (bool *)malloc(_g.nodeCount);
        _g.onlyConnectedFrom(0);
        for (int i = 0; i < g.nodeCount; i++) {
          inAll[i] = true;
        }
        for (int i = g.nodeCount; i < _g.nodeCount; i++) {
          inAll[i] = false;
        }
        TIME(
            pivotPoints =
                nnpnet.run(_g, inAll, nullptr, settings[ARG_EMBEDDING_SIZE]);
            if (settings[ARG_SMOOTH] >= 1) {
              SmoothingFunctions::Laplacian(_g, settings[ARG_SMOOTH]);
            },
            "NNP-NET");

        free(inAll);
      }
      { // First time-step
        Graph<float> _g(g);
        _g.onlyConnectedFrom(0);
        TIME(
            nnpnet.run(_g, pivotPoints, nullptr, settings[ARG_EMBEDDING_SIZE]);
            if (settings[ARG_SMOOTH] >= 1) {
              SmoothingFunctions::Laplacian(_g, settings[ARG_SMOOTH]);
            },
            "NNP-NET");
        std::cout << "Saving to " << outPath << "\n";
        _g.saveToVNA(outPath);
        calculateMetrics(_g, settings);
      }

      int i = 1;
      std::string tsdPath =
          path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
      // Subsequent time-steps
      while (std::filesystem::exists(tsdPath)) {
        g.loadTimeSeries(tsdPath);
        {
          Graph<float> _g(g);
          _g.onlyConnectedFrom(0);
          TIME(
              nnpnet.run(_g, pivotPoints, nullptr,
                         settings[ARG_EMBEDDING_SIZE]);
              if (settings[ARG_SMOOTH] >= 1) {
                SmoothingFunctions::Laplacian(_g, settings[ARG_SMOOTH]);
              },
              "NNP-NET");
          std::cout << "Saving to " << outPath << "\n";
          _g.saveToVNA(outPath.substr(0, outPath.size() - 4) +
                       std::to_string(i) + ".vna");
          calculateMetrics(_g, settings);
        }

        i++;
        tsdPath = path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
      }
      free(pivotPoints);
    } else {
      // Not time-series
      g.loadFromFile(path, settings[ARG_IGNORE_WEIGHTS] > 0);
      g.fillNodeNames();
      g.onlyConnectedFrom(0);
      if (!g.checkFullyConnected()) {
        std::cout << "Not fully connected\n";
      }
      TIME(
          nnpnet.run(g, nullptr, settings[ARG_EMBEDDING_SIZE]);
          if (settings[ARG_SMOOTH] >= 1) {
            SmoothingFunctions::Laplacian(g, settings[ARG_SMOOTH]);
          },
          "NNP-NET");
      std::cout << "Saving to " << outPath << "\n";
      g.saveToVNA(outPath);

      calculateMetrics(g, settings);
      currentStats.fallback = nnpnet.fallbackFlag;
    }

  } break;
  case METHOD_PMDS: {
    if (settings[ARG_TIME_SERIES]) {
      Graph<T> g(settings[ARG_DIMENSIONS]);
      g.loadFromFile(path, settings[ARG_IGNORE_WEIGHTS] > 0);
      g.fillNodeNames();
      // Check which nodes are in all time series

      PivotMDS<T> pmds;
      pmds.setNumberOfPivots(settings[ARG_PMDS_PIVOTS]);
      int *pivotPoints;

      { // Get pivot points that are in all time-steps
        int i = 1;
        Graph<T> _g(g);
        std::string tsdPath =
            path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
        while (std::filesystem::exists(tsdPath)) {
          _g.loadTimeSeries(tsdPath, true);
          i++;
          tsdPath =
              path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
        }

        Graph<T> __g(g);
        __g.onlyConnectedFrom(0);
        bool *inAll = (bool *)malloc(_g.nodeCount * sizeof(bool));

        for (int j = 0; j < _g.nodeCount; j++) {
          inAll[j] = __g.namesToId.count(j) > 0;
        }

        TIME(pivotPoints = pmds.call(_g, inAll), "PMDS");

        free(inAll);
      }

      { // First time-step
        Graph<T> _g(g);
        _g.onlyConnectedFrom(0);
        TIME(pmds.call(_g, pivotPoints), "PMDS");
        std::cout << "Saving to " << outPath << "\n";
        _g.saveToVNA(outPath);
        calculateMetrics(_g, settings);
      }

      int i = 1;
      std::string tsdPath =
          path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
      // All subsequent time-steps
      while (std::filesystem::exists(tsdPath)) {
        g.loadTimeSeries(tsdPath);
        Graph<T> _g(g);
        _g.onlyConnectedFrom(0);

        pmds.setNumberOfPivots(settings[ARG_PMDS_PIVOTS]);
        TIME(pmds.call(_g, pivotPoints), "PMDS");
        std::cout << "Saving to " << outPath << "\n";
        _g.saveToVNA(outPath.substr(0, outPath.size() - 4) + std::to_string(i) +
                     ".vna");
        calculateMetrics(_g, settings);

        i++;
        tsdPath = path.substr(0, path.size() - 4) + std::to_string(i) + ".tsd";
      }

      free(pivotPoints);
    } else { // PMDS non time series
      Graph<T> g(settings[ARG_DIMENSIONS]);
      g.loadFromFile(path, settings[ARG_IGNORE_WEIGHTS] > 0);
      PivotMDS<T> pmds;
      pmds.setNumberOfPivots(settings[ARG_PMDS_PIVOTS]);
      TIME(pmds.call(g), "PMDS");
      std::cout << "Saving to " << outPath << "\n";
      g.saveToVNA(outPath);
      calculateMetrics(g, settings);
    }
  } break;
  case METHOD_TSNET: {
    Graph<double> g(settings[ARG_DIMENSIONS]);
    g.loadFromFile(path, settings[ARG_IGNORE_WEIGHTS] > 0);
    TSNET<double> tsnet;
    tsnet.perp = settings[ARG_PERP];
    tsnet.featureWeight = settings[ARG_FEATURE_WEIGHT];
    TIME(tsnet.tsNET(g, settings[ARG_THETA]), "tsNET");
    std::cout << "Saving to " << outPath << "\n";
    g.saveToVNA(outPath);
    calculateMetrics(g, settings);
  } break;
  case METHOD_TSNETSTAR: {
    Graph<double> g(settings[ARG_DIMENSIONS]);
    g.loadFromFile(path, settings[ARG_IGNORE_WEIGHTS] > 0);
    TSNET<double> tsnet;
    tsnet.perp = settings[ARG_PERP];
    tsnet.featureWeight = settings[ARG_FEATURE_WEIGHT];
    TIME(tsnet.tsNETStar(g, settings[ARG_THETA]), "tsNET*");
    std::cout << "Saving to " << outPath << "\n";
    g.saveToVNA(outPath);
    calculateMetrics(g, settings);
  } break;
  }
}

void printSettings(std::unordered_map<Argument, double> &settings) {
  std::cout << "Output dimensions: " << settings[ARG_DIMENSIONS] << "\n";
  std::cout << "Method used: ";
  switch ((Method)(int)settings[ARG_METHOD]) {
  case METHOD_TSNET:
    std::cout << "tsNET\n";
    std::cout << "tsNET* settings: \n"
              << "\tPerplexity: " << settings[ARG_PERP] << "\n"
              << "\tTheta: " << settings[ARG_THETA] << "\n"
              << "\tMultivariate Feature Weight: "
              << settings[ARG_FEATURE_WEIGHT] << "\n";
    break;
  case METHOD_TSNETSTAR:
    std::cout << "tsNET*\n";
    std::cout << "tsNET* settings: \n"
              << "\tPerplexity: " << settings[ARG_PERP] << "\n"
              << "\tTheta: " << settings[ARG_THETA] << "\n"
              << "\tMultivariate Feature Weight: "
              << settings[ARG_FEATURE_WEIGHT] << "\n";
    break;
  case METHOD_NNPNET:
    std::cout << "NNP-NET\n";
    std::cout << "NNP-NET settings: \n"
              << "\tPerplexity: " << settings[ARG_PERP] << "\n"
              << "\tSubgraph size: " << (int)settings[ARG_SUBGRAPH_SIZE] << "\n"
              << "\tEmbedding size: " << (int)settings[ARG_EMBEDDING_SIZE]
              << "\n"
              << "\tTheta: " << settings[ARG_THETA] << "\n"
              << "\tSmoothing passes: " << settings[ARG_SMOOTH] << "\n"
              << "\tPMDS Pivots: " << settings[ARG_PMDS_PIVOTS] << "\n"
              << "\tBatch size: " << settings[ARG_BATCH_SIZE] << "\n"
              << "\tTraining epochs: " << settings[ARG_TRAINING_EPOCHS] << "\n"
              << "\tUses gpu: " << (settings[ARG_GPU] == 0 ? "No" : "Yes")
              << "\n"
              << "\tUses float precision: "
              << (settings[ARG_USE_FLOAT] == 0 ? "Double" : "Float") << "\n"
              << "\tTime series data: "
              << (settings[ARG_TIME_SERIES] == 0 ? "No" : "Yes") << "\n"
              << "\tMultivariate Feature Weight: "
              << settings[ARG_FEATURE_WEIGHT] << "\n";

    break;
  case METHOD_PMDS:
    std::cout << "PMDS\n";
    std::cout << "tsNET* settings: \n"
              << "\tPMDS Pivots: " << settings[ARG_PMDS_PIVOTS] << "\n"
              << "\tUses float precision: "
              << (settings[ARG_USE_FLOAT] == 0 ? "Double" : "Float") << "\n"
              << "\tTime series data: "
              << (settings[ARG_TIME_SERIES] == 0 ? "No" : "Yes") << "\n";
    break;
  }

  std::cout << "\nCalculate Metrics:\n"
            << "\tStress: "
            << (settings[ARG_CALC_STRESS] == 0 ? "No\n" : "Yes\n")
            << "\tFeature Stress: "
            << (settings[ARG_CALC_FEATURE_STRESS] == 0 ? "No\n" : "Yes\n")
            << "\tNeighborhood preservation: "
            << (settings[ARG_CALC_NP] == 0 ? "No\n\n" : "Yes\n\n");
}

int main(int argc, char *argv[]) {
  py::scoped_interpreter guard{};
  Threadpool::initialize();

  std::string path;
  std::string outPath;

  std::unordered_map<Argument, double> settings;
  settings[ARG_SMOOTH] = 3;
  settings[ARG_THETA] = 0.25;
  settings[ARG_GPU] = 0;
  settings[ARG_PERP] = 40;
  settings[ARG_METHOD] = METHOD_NNPNET;
  settings[ARG_EMBEDDING_SIZE] = 50;
  settings[ARG_SUBGRAPH_SIZE] = 10000;
  settings[ARG_PMDS_PIVOTS] = 250;
  settings[ARG_DIMENSIONS] = 2;
  settings[ARG_BATCH_SIZE] = 64;
  settings[ARG_TRAINING_EPOCHS] = 40;
  settings[ARG_USE_FLOAT] = 0;
  settings[ARG_TIME_SERIES] = 0;
  settings[ARG_FEATURE_WEIGHT] = 0.5;
  settings[ARG_CALC_FEATURE_STRESS] = false;
  settings[ARG_CALC_STRESS] = false;
  settings[ARG_CALC_NP] = false;
  settings[ARG_IGNORE_WEIGHTS] = false;

  if (argc == 1) {
    std::cout << "Enter path\n";
    std::cin >> path;
  } else {
    int i = 1;
    if (argMap.count(argv[1]) == 0) {
      path = argv[1];
      i++;
    } else {
      std::cout << "Enter path\n";
      std::cin >> path;
    }
    for (; i < argc; i++) {
      if (argMap.count(argv[i]) == 0) {
        std::cout << "Invalid argument " << argv[i] << "\n";
        continue;
      }
      Argument setting = argMap[argv[i]];
      i++;
      if (i == argc || argMap.count(argv[i]) > 0) {
        i--;
        std::cout << "Value missing for argument " << argv[i] << "\n";
        continue;
      }
      std::string argS = std::string(argv[i]);
      switch (setting) {
      case ARG_SMOOTH:
      case ARG_THETA:
      case ARG_PERP:
      case ARG_EMBEDDING_SIZE:
      case ARG_SUBGRAPH_SIZE:
      case ARG_DIMENSIONS:
      case ARG_BATCH_SIZE:
      case ARG_TRAINING_EPOCHS:
      case ARG_PMDS_PIVOTS:
      case ARG_FEATURE_WEIGHT:
        settings[setting] = std::stod(argv[i]);
        break;
      case ARG_OUTPATH:
        outPath = argv[i];
        break;
      case ARG_GPU:
      case ARG_TIME_SERIES:
      case ARG_USE_FLOAT:
      case ARG_CALC_STRESS:
      case ARG_CALC_FEATURE_STRESS:
      case ARG_CALC_NP:
      case ARG_IGNORE_WEIGHTS:
        settings[setting] =
            (argS == "t" || argS == "true" || argS == "T" || argS == "True" ||
             argS == "TRUE" || argS == "y" || argS == "Y" || argS == "yes" ||
             argS == "Yes" || argS == "YES" || argS == "1")
                ? 1
                : 0;
        break;
      case ARG_METHOD:
        if (argS == "tsNET" || argS == "tsnet" || argS == "TSNET") {
          settings[setting] = METHOD_TSNET;
        } else if (argS == "tsNET*" || argS == "tsnet*" || argS == "TSNET*") {
          settings[setting] = METHOD_TSNETSTAR;
        } else if (argS == "pmds" || argS == "PMDS" || argS == "Pmds" ||
                   argS == "PivotMDS" || argS == "pivotmds" ||
                   argS == "pivotMDS") {
          settings[setting] = METHOD_PMDS;
        } else if (argS == "NNP-NET" || argS == "NNPNET" || argS == "nnp-net" ||
                   argS == "NNP-net" || argS == "nnp-NET" || argS == "nnpnet" ||
                   argS == "nnpNet" || argS == "nnpNET") {
          settings[setting] = METHOD_NNPNET;
        } else {
          std::cout << "Embedding method " << argv[i]
                    << " not recognized, defaulting to NNP-NET. Use "
                       "either:\nNNP-NET\nPMDS\ntsNET\ntsNET*\n\n";
        }
        break;
      }
    }
  }

  printSettings(settings);

  std::string csvPath =
      outPath == "" ? "results.csv" : outPath + "/results.csv";
  if (outPath != "" && outPath[outPath.size() - 1] != '/' &&
      outPath[outPath.size() - 1] != '\\' && path[path.size() - 4] != '.') {
    csvPath = outPath + "/results.csv";
  } else if (path[path.size() - 4] == '.') {
    csvPath = path.substr(0, path.find_last_of('.')) + "_results.csv";
  }

  if (path[path.size() - 4] == '.') {
    fs::path inP(path);
    NNPNet::StatsCollector::clear();
    currentStats = RunStats();
    currentStats.graphName = inP.stem().string();

    if (outPath == "") {
      outPath = path.substr(0, path.size() - 4) + "_out.vna";
    }
    if (!(outPath[outPath.size() - 3] == 'v' &&
          outPath[outPath.size() - 2] == 'n' &&
          outPath[outPath.size() - 1] == 'a')) {
      outPath += ".vna";
    }

    try {
      if (settings[ARG_USE_FLOAT]) {
        createLayoutFor<float>(path, outPath, settings);
      } else {
        createLayoutFor<double>(path, outPath, settings);
      }
    } catch (const std::exception &e) {
      currentStats.success = false;
      currentStats.errorMessage = e.what();
      std::cout << "Error: " << e.what() << "\n";
    }
    appendToCSV(csvPath, currentStats);
  } else {
    for (const auto &entry : fs::directory_iterator(path)) {
      if (entry.is_directory())
        continue;
      std::string p = entry.path().string();
      auto splittedPath = Utils::split(p, '\\');
      if (splittedPath.size() == 1) {
        splittedPath = Utils::split(p, '/');
      }
      std::string fileName = splittedPath[splittedPath.size() - 1];
      if (entry.path().extension() != ".mtx")
        continue;

      std::cout << "\nCurrent file: " << p << "\n";

      NNPNet::StatsCollector::clear();
      currentStats = RunStats();
      currentStats.graphName = entry.path().stem().string();

      std::string oPath;
      if (outPath == "") {
        oPath = p.substr(0, p.size() - 4) + "_out.vna";
      } else {
        oPath = outPath;
        if (oPath[oPath.size() - 1] != '\\' && oPath[oPath.size() - 1] != '/') {
          oPath += '/';
        }
        oPath += fileName.substr(0, fileName.length() - 4) + ".vna";
      }

      try {
        if (settings[ARG_USE_FLOAT]) {
          createLayoutFor<float>(p, oPath, settings);
        } else {
          createLayoutFor<double>(p, oPath, settings);
        }
      } catch (const std::exception &e) {
        currentStats.success = false;
        currentStats.errorMessage = e.what();
        std::cout << "Error processing " << currentStats.graphName << ": "
                  << e.what() << "\n";
      } catch (...) {
        currentStats.success = false;
        currentStats.errorMessage = "Unknown error";
        std::cout << "Unknown error processing " << currentStats.graphName
                  << "\n";
      }
      appendToCSV(csvPath, currentStats);
    }
  }

  return 0;
}

// Python module functions

static std::unordered_map<Argument, double> currentSettings;

void init(int smoothing = 3, double theta = 0.25, bool gpu = false,
          int perplexity = 40, int embeddingSize = 50, int subgraphSize = 10000,
          int pmdsPivots = 250, int outputDimensions = 2, int batchSize = 64,
          int trainingEpochs = 40) {
  currentSettings[ARG_SMOOTH] = smoothing;
  currentSettings[ARG_THETA] = theta;
  currentSettings[ARG_GPU] = gpu;
  currentSettings[ARG_PERP] = perplexity;
  currentSettings[ARG_METHOD] = METHOD_NNPNET;
  currentSettings[ARG_EMBEDDING_SIZE] = embeddingSize;
  currentSettings[ARG_SUBGRAPH_SIZE] = subgraphSize;
  currentSettings[ARG_PMDS_PIVOTS] = pmdsPivots;
  currentSettings[ARG_DIMENSIONS] = outputDimensions;
  currentSettings[ARG_BATCH_SIZE] = batchSize;
  currentSettings[ARG_TRAINING_EPOCHS] = trainingEpochs;
  currentSettings[ARG_USE_FLOAT] = 0;       // We will always use doubles
  currentSettings[ARG_TIME_SERIES] = false; // No time series
  currentSettings[ARG_IGNORE_WEIGHTS] = false;
};

void run(std::string path, std::string outPath, int smoothing = 3,
         double theta = 0.25, bool gpu = false, int perplexity = 40,
         int method = METHOD_NNPNET, int embeddingSize = 50,
         int subgraphSize = 10000, int pmdsPivots = 250,
         int outputDimensions = 2, int batchSize = 64, int trainingEpochs = 40,
         bool useFloat = true, bool timeSeriesData = false) {
  std::unordered_map<Argument, double> settings;
  settings[ARG_SMOOTH] = smoothing;
  settings[ARG_THETA] = theta;
  settings[ARG_GPU] = gpu;
  settings[ARG_PERP] = perplexity;
  settings[ARG_METHOD] = method;
  settings[ARG_EMBEDDING_SIZE] = embeddingSize;
  settings[ARG_SUBGRAPH_SIZE] = subgraphSize;
  settings[ARG_PMDS_PIVOTS] = pmdsPivots;
  settings[ARG_DIMENSIONS] = outputDimensions;
  settings[ARG_BATCH_SIZE] = batchSize;
  settings[ARG_TRAINING_EPOCHS] = trainingEpochs;
  settings[ARG_USE_FLOAT] = useFloat;
  settings[ARG_TIME_SERIES] = timeSeriesData;

  printSettings(settings);

  if (path[path.size() - 4] == '.') {
    if (outPath == "") {
      outPath = path.substr(0, path.size() - 4) + "_out.vna";
    }
    if (!(outPath[outPath.size() - 3] == 'v' &&
          outPath[outPath.size() - 2] == 'n' &&
          outPath[outPath.size() - 1] == 'a')) {
      outPath += ".vna";
    }
    if (settings[ARG_USE_FLOAT]) {
      createLayoutFor<float>(path, outPath, settings);
    } else {
      createLayoutFor<double>(path, outPath, settings);
    }

  } else {
    for (const auto &entry : fs::directory_iterator(path)) {
      std::string p = entry.path().string();
      auto splittedPath = Utils::split(p, '\\');
      if (splittedPath.size() == 1) {
        splittedPath = Utils::split(p, '/');
      }
      std::string fileName = splittedPath[splittedPath.size() - 1];

      std::cout << "\nCurrent file: " << p << "\n";

      std::string oPath;
      if (outPath == "") {
        oPath = p.substr(0, p.size() - 4) + "_out.vna";
      } else {
        oPath = outPath;
        if (oPath[oPath.size() - 1] != '\\' || oPath[oPath.size() - 1] != '/') {
          oPath += '/';
        }
        oPath += fileName.substr(0, fileName.length() - 4) + ".vna";
      }

      if (settings[ARG_USE_FLOAT]) {
        createLayoutFor<float>(p, oPath, settings);
      } else {
        createLayoutFor<double>(p, oPath, settings);
      }
    }
  }
};

Graph<double> *loadFile(std::string path) {
  Graph<double> *g = new Graph<double>(currentSettings[ARG_DIMENSIONS]);
  g->loadFromFile(path, currentSettings[ARG_IGNORE_WEIGHTS] > 0);
  return g;
};

void saveFile(std::string path, Graph<double> *graph) {
  graph->saveToVNA(path);
};

void createLayoutFor(Graph<double> *_g,
                     std::unordered_map<Argument, double> &settings) {
  printSettings(settings);
  switch ((Method)(int)settings[ARG_METHOD]) {
  case METHOD_NNPNET: {
    Graph<float> g(*_g);
    NNPNET nnpnet;
    nnpnet.perplexity = settings[ARG_PERP];
    nnpnet.gpu = settings[ARG_GPU] > 0;
    nnpnet.useFloats = settings[ARG_USE_FLOAT] > 0;
    nnpnet.theta = settings[ARG_THETA];
    nnpnet.subgraphPoints = settings[ARG_SUBGRAPH_SIZE];
    nnpnet.pmdsPivots = settings[ARG_PMDS_PIVOTS];
    nnpnet.trainingEpochs = settings[ARG_TRAINING_EPOCHS];
    nnpnet.batchSize = settings[ARG_BATCH_SIZE];

    g.fillNodeNames();
    g.onlyConnectedFrom(0);
    if (!g.checkFullyConnected()) {
      std::cout << "Not fully connected\n";
    }
    TIME(
        nnpnet.run(g, nullptr, settings[ARG_EMBEDDING_SIZE]);
        if (settings[ARG_SMOOTH] >= 1) {
          SmoothingFunctions::Laplacian(g, settings[ARG_SMOOTH]);
        },
        "NNP-NET");

    for (unsigned long long i = 0; i < _g->nodeCount * _g->outputDim; i++) {
      _g->Y[i] = g.Y[i];
    }

  } break;
  case METHOD_PMDS: {
    PivotMDS<double> pmds;
    pmds.setNumberOfPivots(settings[ARG_PMDS_PIVOTS]);
    TIME(pmds.call(*_g), "PMDS");

  } break;
  case METHOD_TSNET: {
    TSNET<double> tsnet;
    tsnet.perp = settings[ARG_PERP];
    TIME(tsnet.tsNET(*_g, settings[ARG_THETA]), "tsNET");
  } break;
  case METHOD_TSNETSTAR: {
    TSNET<double> tsnet;
    tsnet.perp = settings[ARG_PERP];
    TIME(tsnet.tsNETStar(*_g, settings[ARG_THETA]), "tsNET*");
  } break;
  }
}

void runNNPNET(Graph<double> *graph) {
  currentSettings[ARG_METHOD] = METHOD_NNPNET;
  createLayoutFor(graph, currentSettings);
}
void runPMDS(Graph<double> *graph) {
  currentSettings[ARG_METHOD] = METHOD_PMDS;
  createLayoutFor(graph, currentSettings);
}
void runTSNET(Graph<double> *graph) {
  currentSettings[ARG_METHOD] = METHOD_TSNET;
  createLayoutFor(graph, currentSettings);
}
void runTSNETSTAR(Graph<double> *graph) {
  currentSettings[ARG_METHOD] = METHOD_TSNETSTAR;
  createLayoutFor(graph, currentSettings);
}

int getNodeCount(Graph<double> *graph) { return graph->nodeCount; }

py::array_t<double> getEmbedding(Graph<double> *graph) {
  return py::array_t<double>({graph->nodeCount, graph->outputDim}, graph->Y);
}

std::vector<std::vector<Edge<double>>> *getEdges(Graph<double> *graph) {
  return &graph->edges;
}

Graph<double> *initializeGraph(int nodeCount) {
  return new Graph<double>(currentSettings[ARG_DIMENSIONS], nodeCount);
};

void addEdge(Graph<double> *graph, int a, int b, double weight) {
  graph->edges[a].push_back(Edge<double>(b, weight));
  graph->edges[b].push_back(Edge<double>(a, weight));
};

PYBIND11_MODULE(NNP_NET, m /*, py::mod_gil_not_used()*/) {
  m.doc() = "This is the module docs.";

  py::class_<Graph<double>>(m, "Graph");
  py::class_<Edge<double>>(m, "Edge")
      .def_readwrite("Other", &Edge<double>::other)
      .def_readwrite("Weight", &Edge<double>::weight);

  m.def("loadGraph", &loadFile, "path"_a);
  m.def("saveGraph", &saveFile, "path"_a, "graph"_a);

  m.def("initializeGraph", &initializeGraph, "nodeCount"_a);
  m.def("addEdge", &addEdge, "graph"_a, "a"_a, "b"_a, "weight"_a = 1);

  m.def("NNP_NET", &runNNPNET, "graph"_a);
  m.def("PMDS", &runPMDS, "graph"_a);
  m.def("tsNET", &runTSNET, "graph"_a);
  m.def("tsNETStar", &runTSNETSTAR, "graph"_a);

  m.def("getNodeCount", &getNodeCount, "graph"_a);
  m.def("getEmbedding", &getEmbedding, "graph"_a);
  m.def("getEdges", &getEdges, "graph"_a);
  m.def("initialize", &init, "smoothing"_a = 3, "theta"_a = 0.25,
        "gpu"_a = false, "perplexity"_a = 40, "embeddingSize"_a = 50,
        "subgraphSize"_a = 10000, "pmdsPivots"_a = 250,
        "outputDimensions"_a = 2, "batchSize"_a = 64, "trainingEpochs"_a = 40);
  m.def("runAll", &run, "path"_a, "outPath"_a = "", "smoothing"_a = 3,
        "theta"_a = 0.25, "gpu"_a = false, "perplexity"_a = 40, "method"_a = 0,
        "embeddingSize"_a = 50, "subgraphSize"_a = 10000, "pmdsPivots"_a = 250,
        "outputDimensions"_a = 2, "batchSize"_a = 64, "trainingEpochs"_a = 40,
        "useFloat"_a = true, "timeSeriesData"_a = false);
};