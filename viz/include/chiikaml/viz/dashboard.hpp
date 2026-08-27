#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// Forward-declaration : evite d'inclure httplib.h (un gros header)
// dans l'interface publique de Dashboard. Seul dashboard.cpp a
// besoin de connaitre httplib::Server en detail.
namespace httplib {
class Server;
}

namespace chiikaml::viz {

// Dashboard local : sert une petite page web (le frontend React deja
// compile) et lui pousse des mises a jour en direct via
// Server-Sent Events (SSE).
//
// SSE plutot que WebSocket pour cette v1 : on n'a besoin que d'un
// flux dans un seul sens (le programme C++ -> le navigateur, jamais
// l'inverse), et SSE est une simple reponse HTTP en streaming, bien
// plus simple a mettre en oeuvre correctement qu'une vraie connexion
// WebSocket (poignee de main, format de trame binaire...). Un futur
// besoin bidirectionnel (le navigateur qui renvoie des commandes au
// programme C++) justifierait de migrer vers WebSocket plus tard.
//
// Design "un seul serveur partage" : Dashboard::instance() renvoie
// toujours la MEME instance (singleton), pour que plusieurs appels a
// visualize() depuis differents endroits du programme (ou differents
// modeles) affichent tous sur la meme page web deja ouverte, plutot
// que d'ouvrir un nouveau serveur/onglet a chaque fois.
class Dashboard {
public:
    static Dashboard& instance();

    Dashboard(const Dashboard&) = delete;
    Dashboard& operator=(const Dashboard&) = delete;

    // Demarre le serveur au tout premier appel (threads-safe : les
    // appels suivants ne redemarrent rien) et ouvre le navigateur par
    // defaut sur cette page. `payload_json` est la donnee affichee
    // immediatement. `frontend_dir` doit pointer vers le build compile
    // du frontend React (viz/frontend/dist).
    void show(const std::string& payload_json, int port = 8787,
              const std::string& frontend_dir = "viz/frontend/dist");

    // Pousse une nouvelle donnee vers la page deja ouverte -- aucun
    // nouvel onglet, la page existante se met a jour en direct.
    // Suppose que show() a deja ete appele au moins une fois.
    void push(const std::string& payload_json);

private:
    Dashboard() = default;
    ~Dashboard();

    void run_server(int port, std::string frontend_dir);
    static void open_browser(const std::string& url);

    std::mutex mutex_;
    std::condition_variable cv_;
    std::string latest_payload_;
    std::uint64_t version_ = 0;
    bool started_ = false;

    std::unique_ptr<httplib::Server> server_;
    std::thread server_thread_;
};

} // namespace chiikaml::viz
