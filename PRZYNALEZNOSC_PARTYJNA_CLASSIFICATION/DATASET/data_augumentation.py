text = '''
wysoka izbo, ta ustawa to jest absolutne minimum bezpieczeństwa i to jest ustawa, za którą naprawdę może zagłosować każdy uczciwy konserwatysta, który będzie się z nami, z lewicą spierać o wartości, o rozwiązania, o praktykę. tutaj mówimy po prostu o bezpieczeństwie. tutaj mówimy o rozwiązaniu, które ma skończyć z sytuacją, w której polskie państwo używa narzędzi karnych, żeby ścigać swoich własnych obywateli w sytuacji, kiedy tego robić po prostu nie powinno. mój apel jest taki: drodzy państwo, nie zacinajcie się, spójrzcie na tekst tej ustawy i zagłosujcie rozumnie, po prostu zagłosujcie za tą ustawą. dziękuję.
'''

import argostranslate.package
argostranslate.package.update_package_index()

available_packages = argostranslate.package.get_available_packages()
# Filter for pl→en and en→pl
pl_en_pkg = next(
    pkg for pkg in available_packages if pkg.from_code == "pl" and pkg.to_code == "en"
)
en_pl_pkg = next(
    pkg for pkg in available_packages if pkg.from_code == "en" and pkg.to_code == "pl"
)
argostranslate.package.install_from_path(pl_en_pkg.download())
argostranslate.package.install_from_path(en_pl_pkg.download())

import argostranslate.translate

def backtranslate(text: str) -> (str, str):
    # Polish → English
    english = argostranslate.translate.translate(text, "pl", "en")
    # English → Polish
    back_polish = argostranslate.translate.translate(english, "en", "pl")
    return english, back_polish

if __name__ == "__main__":
    en, pl_back = backtranslate(text)
    print("=== Intermediate English Translation ===\n", en)
    print("\n=== Final Back-Translated Polish ===\n", pl_back)
