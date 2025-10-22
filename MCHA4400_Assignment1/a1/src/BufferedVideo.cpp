#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "BufferedVideo.h"

BufferedVideoReader::BufferedVideoReader(int bufferSize)
    : readBufferSize(bufferSize)
    , readFinished(false)
    , pCap(nullptr)
{}

BufferedVideoReader::~BufferedVideoReader()
{
    if (pCap)
        pCap->release();
}

void BufferedVideoReader::start(cv::VideoCapture & cap)
{
    pCap = &cap;
    videoReaderThread = std::thread(&BufferedVideoReader::videoReader, this, std::ref(cap));
}

void BufferedVideoReader::stop()
{
    readFinished = true;
    readBufferNotFull.notify_one(); // unblock waiting thread
    videoReaderThread.join();
    pCap->release();
    pCap = nullptr;
}

cv::Mat BufferedVideoReader::read()
{
    cv::Mat frame;
    {   // begin scope for lock
        std::unique_lock<std::mutex> lock(readBufferMutex);

        // Wait until buffer not empty or video finished
        while (readBuffer.empty() && !readFinished)
            readBufferNotEmpty.wait(lock);

        if (readBuffer.empty() && readFinished)
            return frame;

        frame = readBuffer.front();
        readBuffer.pop();
    }   // end scope for lock
    readBufferNotFull.notify_one();
    return frame;
}

void BufferedVideoReader::videoReader(cv::VideoCapture & cap)
{
    readFinished = false;
    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        {   // begin scope for lock
            std::unique_lock<std::mutex> lock(readBufferMutex);

            // Wait until buffer not full or video finished
            while (readBuffer.size() >= readBufferSize && !readFinished)
                readBufferNotFull.wait(lock);

            if (readFinished)
                break;

            readBuffer.emplace(frame);
        }   // end scope for lock
        readBufferNotEmpty.notify_one();
    }
    readFinished = true;
    readBufferNotEmpty.notify_one(); // unblock waiting thread
}

BufferedVideoWriter::BufferedVideoWriter(int bufferSize)
    : writeBufferSize(bufferSize)
    , writeFinished(false)
    , pVideo(nullptr)
{}

BufferedVideoWriter::~BufferedVideoWriter()
{
    if (pVideo)
        pVideo->release();
}

void BufferedVideoWriter::start(cv::VideoWriter & video)
{
    pVideo = &video;
    videoWriterThread = std::thread(&BufferedVideoWriter::videoWriter, this, std::ref(video));
}

void BufferedVideoWriter::stop()
{
    writeFinished = true;
    writeBufferNotEmpty.notify_one(); // unblock waiting thread
    videoWriterThread.join();
    pVideo->release();
    pVideo = nullptr;
}

void BufferedVideoWriter::write(const cv::Mat & frame)
{
    {   // begin scope for lock
        std::unique_lock<std::mutex> lock(writeBufferMutex);

        // Wait until buffer not full
        while (writeBuffer.size() >= writeBufferSize)
            writeBufferNotFull.wait(lock);

        writeBuffer.emplace(frame.clone());
    }   // end scope for lock
    writeBufferNotEmpty.notify_one();
}

void BufferedVideoWriter::videoWriter(cv::VideoWriter & video)
{
    while (true)
    {
        cv::Mat frame;
        {   // begin scope for lock
            std::unique_lock<std::mutex> lock(writeBufferMutex);

            // Wait until buffer not empty or video finished
            while (writeBuffer.empty() && !writeFinished)
                writeBufferNotEmpty.wait(lock);

            if (writeBuffer.empty() && writeFinished)
                break;

            frame = writeBuffer.front();
            writeBuffer.pop();
        }   // end scope for lock
        writeBufferNotFull.notify_one();
        video.write(frame);
    }
}
