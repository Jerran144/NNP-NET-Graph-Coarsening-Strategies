#pragma once

#include <thread>
#include <mutex>
#include <functional>
#include <vector>
#include <condition_variable>

#include <iostream>

namespace NNPNet {

	class ThreadpoolThread {
	public:

		ThreadpoolThread() {
			thread = new std::thread(&ThreadpoolThread::entry, this);
		};

		~ThreadpoolThread() {
			stop = true;
			cVar.notify_one();
			thread->join();
			delete thread;
		}

		void doJob(std::function<void(void)> j) {
			job = j;
			done = false;
			cVar.notify_all();
		};

		void setJob(std::function<void(void)> j) {
			job = j;
		};

		void redoJob() {
			done = false;
			cVar.notify_all();
		};

		void join() {
			std::unique_lock<std::mutex> l(lock);
			if (done) return;
			cVar.wait(l);
		};

	private:
		void entry() {
			while (!stop) {
				{
					std::unique_lock<std::mutex> l(lock);
					done = true;
					cVar.notify_all();
					cVar.wait(l);
				}
				if (stop)
				{
					return;
				}
				job();

			}
		};

		volatile bool stop = false;
		volatile bool done = false;

		std::thread* thread;
		std::mutex lock;
		std::condition_variable cVar;
		std::function<void(void)> job = nullptr;
	};

	class Threadpool {
	public:

		~Threadpool() {
			for (auto tt : threads) {
				delete tt;
			}
		}

		static void initialize(int tCount = -1) {
			if (tCount == -1) {
				tCount = std::thread::hardware_concurrency();
			}
			t = new Threadpool(tCount);
		}

		template<class T>
		static T divideWork(std::function<T(int, int)> f, int size, std::function<T(T,T)> combine) {
			// Initialize the threadpool if it isn't yet
			if (t == nullptr) {
				initialize();
			}
			// Start up threads
			T* outList = new T[size];
			auto func = [outList, f](int id, int start, int end){ outList[id] = f(start, end); };
			for (int i = 0; i < t->threads.size(); i++) {
				int start = (int)(size * (double)(i) / (double)(t->threads.size()));
				int end = (int)(size * (double)(i + 1) / (double)(t->threads.size()));
				t->threads[i]->doJob(std::bind(func, i, start, end));
			}

			// Combine threads
			T tot;
			for (int i = 0; i < t->threads.size(); i++) {
				t->threads[i]->join();
				if (i == 0) {
					tot = outList[i];
				}
				else {
					tot = combine(tot, outList[i]);
				}
			}
			delete[] outList;
			return tot;
		};

		static void divideWork(std::function<void(int, int)> f, int size) {
			// Initialize the threadpool if it isn't yet
			if (t == nullptr) {
				initialize();
			}
			// Start up threads
			auto func = [](std::function<void(int, int)> ff, int start, int end){ ff(start, end); };
			for (int i = 0; i < t->threads.size(); i++) {
				int start = (int)(size * (double)(i) / (double)(t->threads.size()));
				int end = (int)(size * (double)(i + 1) / (double)(t->threads.size()));
				t->threads[i]->doJob(std::bind(func, f, start, end));
			}

			// Join threads
			for (int i = 0; i < t->threads.size(); i++) {
				t->threads[i]->join();
			}
		};

		static Threadpool* createJob(std::function<void(int, int)> f, int size) {
			int tCount = std::thread::hardware_concurrency();
			Threadpool* jt = new Threadpool(tCount);
			auto func = [](std::function<void(int, int)> ff, int start, int end) { ff(start, end); };
			for (int i = 0; i < jt->threads.size(); i++) {
				int start = (int)(size * (double)(i) / (double)(jt->threads.size()));
				int end = (int)(size * (double)(i + 1) / (double)(jt->threads.size()));
				jt->threads[i]->setJob(std::bind(func, f, start, end));
			}
			return jt;
		}

		void performJobs() {
			for (int i = 0; i < threads.size(); i++) {
				threads[i]->redoJob();
			}
			for (int i = 0; i < threads.size(); i++) {
				threads[i]->join();
			}
		}

	private:

		static Threadpool* t;

		Threadpool(int tCount) {
			for (int i = 0; i < tCount; i++) {
				threads.push_back(new ThreadpoolThread());
			}
		};

		std::vector<ThreadpoolThread*> threads;

	};

};
