// CLI minimaliste pour chiikaml : entraine un KNNClassifier depuis
// un fichier CSV et affiche les predictions sur un second fichier.
//
// Usage :
//   chiikaml_cli --train cli/examples/train.csv --test cli/examples/test.csv --k 3
//
// Convention CSV : dans le fichier d'ENTRAINEMENT, la DERNIERE
// colonne est le label (entier) -- toutes les autres colonnes sont
// des features. Dans le fichier de TEST, toutes les colonnes sont
// des features (pas de colonne label).

#include <CLI/CLI.hpp>
#include <iostream>

#include "chiikaml/knn.hpp"
#include "chiikaml/matrix.hpp"

using namespace chiikaml;

int main(int argc, char** argv) {
    CLI::App app{"chiikaml CLI -- k-NN depuis la ligne de commande"};

    std::string train_path;
    std::string test_path;
    std::size_t k = 3;

    app.add_option("--train", train_path, "CSV d'entrainement (features..., derniere colonne = label)")
        ->required();
    app.add_option("--test", test_path, "CSV de test (features seulement, pas de label)")->required();
    app.add_option("--k", k, "Nombre de voisins consideres")->default_val(3);

    CLI11_PARSE(app, argc, argv);

    // TODO(toi), etape par etape :
    //
    // 1) Charge le CSV d'entrainement en une seule Matrix avec
    //    Matrix::from_csv(train_path) -- elle contient encore la
    //    colonne label melangee avec les features a ce stade.
    //
    // 2) Separe cette Matrix en deux :
    //    - X_train : une Matrix(train_data.rows(), train_data.cols() - 1),
    //      remplie avec toutes les colonnes SAUF la derniere (meme
    //      technique de copie ligne/colonne que dans
    //      DecisionTreeClassifier::build() pour construire X_left/X_right)
    //    - y_train : un std::vector<int>, un element par ligne, lu
    //      dans la DERNIERE colonne de train_data
    //      (train_data(i, train_data.cols() - 1)), converti en int
    //      avec static_cast<int>(...)
    //
    // 3) Charge le CSV de test avec Matrix::from_csv(test_path) --
    //    directement utilisable comme X_test, pas de colonne label a
    //    retirer cette fois.
    //
    // 4) Cree un KNNClassifier(k), appelle fit(X_train, y_train),
    //    puis predict(X_test).
    //
    // 5) Affiche chaque prediction sur sa propre ligne (std::cout).

    Matrix train_data = Matrix::from_csv(train_path);
    Matrix X_train(train_data.rows(), train_data.cols() - 1);
    std::vector<int> y_train(train_data.rows());

    Matrix test_data = Matrix::from_csv(test_path);
    Matrix X_test(test_data.rows(), test_data.cols());

    for (std::size_t i = 0; i<train_data.rows(); ++i) {
        for (std::size_t j= 0; j<train_data.cols() - 1; ++j) {
            X_train(i, j) = train_data(i,j);
        }
        y_train[i] = static_cast<int>(train_data(i, train_data.cols() - 1));
    }

    for (std::size_t i = 0; i<test_data.rows(); ++i) {
        for (std::size_t j= 0; j<test_data.cols(); ++j) {
            X_test(i, j) = test_data(i,j);
        }
    }

    KNNClassifier knn(k);
    knn.fit(X_train, y_train);
    std::vector<int> predictions = knn.predict(X_test);
    for (const auto& pred : predictions) {
        std::cout << pred << "\n";
    }

    return 0;
}
