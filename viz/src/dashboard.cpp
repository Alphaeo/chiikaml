#include "chiikaml/viz/dashboard.hpp"

#include <httplib.h>

#include <cstdlib>

namespace chiikaml::viz {

Dashboard& Dashboard::instance() {
    static Dashboard dashboard;
    return dashboard;
}

Dashboard::~Dashboard() {
    if (server_) {
        server_->stop();
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void Dashboard::open_browser(const std::string& url) {
#if defined(_WIN32)
    std::system(("start \"\" \"" + url + "\"").c_str());
#elif defined(__APPLE__)
    std::system(("open \"" + url + "\"").c_str());
#else
    std::system(("xdg-open \"" + url + "\"").c_str());
#endif
}

void Dashboard::run_server(int port, std::string frontend_dir) {
    server_ = std::make_unique<httplib::Server>();
    server_->set_mount_point("/", frontend_dir);

    server_->Get("/events", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        auto last_seen = std::make_shared<std::uint64_t>(0);
        res.set_chunked_content_provider("text/event-stream",
                                          [this, last_seen](std::size_t, httplib::DataSink& sink) {
                                              std::unique_lock<std::mutex> lock(mutex_);
                                              cv_.wait(lock, [this, last_seen] { return version_ != *last_seen; });
                                              std::string payload = latest_payload_;
                                              *last_seen = version_;
                                              lock.unlock();

                                              std::string chunk = "data: " + payload + "\n\n";
                                              sink.write(chunk.data(), chunk.size());
                                              return true;
                                          });
    });

    server_->listen("0.0.0.0", port);
}

void Dashboard::show(const std::string& payload_json, int port, const std::string& frontend_dir) {
    bool need_start = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        need_start = !started_;
        started_ = true;
    }

    if (need_start) {
        server_thread_ = std::thread(&Dashboard::run_server, this, port, frontend_dir);
        // Laisse au serveur le temps de demarrer avant d'ouvrir le
        // navigateur -- approximatif mais suffisant pour une v1
        // (une vraie synchronisation attendrait un signal explicite
        // du serveur, ex: une callback "on bind reussi").
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        open_browser("http://localhost:" + std::to_string(port));
    }

    push(payload_json);
}

void Dashboard::push(const std::string& payload_json) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_payload_ = payload_json;
    ++version_;
    cv_.notify_all();
}

} // namespace chiikaml::viz
