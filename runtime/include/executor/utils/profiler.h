/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system, or translated into any human or computer language in any form by any
 * means, electronic, mechanical, manual or otherwise, or disclosed to third parties without the express written
 * permission of Samsung Electronics. (Use of the Software is restricted to non-commercial, personal or academic,
 * research purpose only)
 */

#ifndef PROFILER_H
#define PROFILER_H

#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

// checked only in linux

// define PROFILING 0 to turn it off
#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) ProfileTimer timer##__LINE__(name)
#ifdef __linux__
#define PROFILE_FUNCTION() PROFILE_SCOPE(__PRETTY_FUNCTION__)
#endif
#ifdef _WIN32
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNSIG__)
#endif

struct ProfileResult {
    std::string name;
    long long start, end;
    uint32_t threadID;
};

struct ProfileEvent {
    std::string cat;
    long long dur;
    std::string name;
    std::string origin_name;
    std::string ph;
    uint32_t pid;
    uint32_t tid;
    long long ts;
};

class ProfileWriter
{
    friend class ProfileTimer;

   private:
    std::string m_currentSession;
    std::ofstream m_outputStream;
    int m_profileCount;
    std::mutex m_mutex;
    bool m_inSession;

   public:
    std::vector<ProfileEvent> _traceEvents;

    // private constructors:
   private:
    ProfileWriter() : m_profileCount(0), m_inSession(false) {}

    // private functions:
   private:
    template <typename T>
    void beginSessionImpl(T&& name, const char* filepath = "results.json")
    {
        if (m_inSession) {
            std::cout << "WARNING: Profiler session is already started\n";
            return;
        }
        m_outputStream.open(filepath);
        m_inSession = true;
        if (!m_outputStream.is_open()) {
            std::cout << "WARNING: couldn't open or create the file for profiler\n";
            m_inSession = false;
            return;
        }
        writeHeader();
        m_currentSession = std::forward<T>(name);
    }

    void endSessionImpl()
    {
        if (m_inSession) {
            writeFooter();
            m_outputStream.close();
            m_currentSession = "";
            m_profileCount = 0;
            m_inSession = false;
        } else
            std::cout << "WARNING: Profiler session is already finished\n";
    }

    void writeHeader()
    {
        m_outputStream << "{\n\"otherData\": {},\n\"traceEvents\": \n[\n";
        m_outputStream.flush();
    }

    void writeFooter()
    {
        m_outputStream << "]}";
        m_outputStream.flush();
    }

    void writeProfile(ProfileResult&& result)
    {
        if (!m_inSession) return;
        std::lock_guard<std::mutex> locker(m_mutex);

        if (m_profileCount++ > 0) m_outputStream << ",";

        std::string name = std::move(result.name);
        std::string origin_name = name;
        std::replace(origin_name.begin(), origin_name.end(), '"', '\'');
        std::regex pattern("(_)(\\d+)");
        name = std::regex_replace(name, pattern, "");

        m_outputStream << "\t{\n";
        m_outputStream << "\t\"cat\": \"function\",\n";
        m_outputStream << "\t\"dur\": " << (result.end - result.start) << ",\n";
        m_outputStream << "\t\"name\": \"" << name << "\",\n";
        m_outputStream << "\t\"origin_name\": \"" << origin_name << "\",\n";
        m_outputStream << "\t\"ph\": \"X\",\n";
        m_outputStream << "\t\"pid\": 0,\n";
        m_outputStream << "\t\"tid\": " << result.threadID << ",\n";
        m_outputStream << "\t\"ts\": " << result.start << '\n';
        m_outputStream << "\t}\n";

        m_outputStream.flush();
    }

    void saveProfile(ProfileResult&& result)
    {
        if (!m_inSession) return;
        std::lock_guard<std::mutex> locker(m_mutex);

        std::string name = std::move(result.name);
        std::string origin_name = name;
        std::replace(origin_name.begin(), origin_name.end(), '"', '\'');
        std::regex pattern("(_)(\\d+)");
        name = std::regex_replace(name, pattern, "");

        ProfileEvent event;
        event.cat = "function";
        event.dur = result.end - result.start;
        event.name = name;
        event.origin_name = origin_name;
        event.ph = "X";
        event.pid = 0;
        event.tid = result.threadID;
        event.ts = result.start;

        this->_traceEvents.push_back(event);
    }

   public:
    static ProfileWriter& get()
    {
        static ProfileWriter instance;
        return instance;
    }

    void generate_chrome_trace(std::string filename)
    {
        std::ofstream outStream;
        outStream.open(filename);

        outStream << "{\n\"otherData\": {},\n\"traceEvents\": \n[\n";
        outStream.flush();

        for (auto idx = 0; idx < _traceEvents.size(); idx++) {
            auto e = _traceEvents.at(idx);
            outStream << "\t{\n";
            outStream << "\t\"cat\": \"function\",\n";
            outStream << "\t\"dur\": " << (e.dur) << ",\n";
            outStream << "\t\"name\": \"" << e.name << "\",\n";
            outStream << "\t\"origin_name\": \"" << e.origin_name << "\",\n";
            outStream << "\t\"ph\": \"X\",\n";
            outStream << "\t\"pid\": 0,\n";
            outStream << "\t\"tid\": " << e.tid << ",\n";
            outStream << "\t\"ts\": " << e.ts << '\n';
            outStream << "\t}\n";
            if (idx < _traceEvents.size() - 1) {
                outStream << ",\n";
            }
        }

        outStream << "]}";
        outStream.flush();
    }

    // public constructors and assignment operators are deleted
   public:
    ProfileWriter(const ProfileWriter& other) = delete;
    ProfileWriter(ProfileWriter&& other) = delete;
    ProfileWriter& operator=(const ProfileWriter& other) = delete;
    ProfileWriter&& operator=(ProfileWriter&& other) = delete;

    ~ProfileWriter()
    {
        if (m_outputStream.is_open()) m_outputStream.close();
    }

   public:
    template <typename T>
    static void beginSession(T&& name, const char* filepath = "results.json")
    {
        static_assert(std::is_constructible_v<std::string, T>);
        get().beginSessionImpl(std::forward<T>(name), filepath);
    }

    static void endSession() { get().endSessionImpl(); }
};

class ProfileTimer
{
   private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTimepoint;
    bool m_stopped;
    std::mutex m_mutex;

   public:
    template <typename T>
    explicit ProfileTimer(T&& name) : m_name(std::forward<T>(name)), m_stopped(false)
    {
        static_assert(std::is_constructible_v<std::string, T>);
        m_startTimepoint = std::chrono::high_resolution_clock::now();
    }

    ~ProfileTimer()
    {
        if (!m_stopped) stop();
    }

    // time_since_epoch()
    void stop()
    {
        auto endTimepoint = std::chrono::high_resolution_clock::now();

        long long start =
            std::chrono::time_point_cast<std::chrono::microseconds>(m_startTimepoint).time_since_epoch().count();
        long long end =
            std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

        uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
        ProfileWriter::get().saveProfile({m_name, start, end, threadID});
        m_stopped = true;
    }
};

#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()

class ProfileWriter
{
   public:
    template <typename T>
    static void beginSession(T&&, const char* = "filename")
    {
    }
    static void endSession() {}
};
#endif
#endif  // PROFILER_H
