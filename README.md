# Your Advisor API

API do rekomendacji porad psychologicznych i doradztwa zawodowego z wykorzystaniem AI. Używane w aplikacji mobilnej Your Advisor, zgłoszonej do konkursu Hack Heroes 2025.

**Produkcyjne API znajduje się pod adresem https://hackheroes25-advice.fly.dev.**

Frontend wraz ze zdjęciami i filmikiem z demem, znajduje się [TUTAJ](https://github.com/Alex09ander/Your-Advisor?tab=readme-ov-file)

## Funkcjonalności

- **Wsparcie psychologiczne dostosowane do użytkownika** - użytkownik przechodzi test, a następnie otrzymuje odpowiedzi dopasowane do jego sytuacji i osobowości. Model AI przeszukuje bazę treści i wybiera te porady, których sens wskazuje na realny związek z problemem zgłoszonym przez użytkownika.
- **Doradztwo zawodowe oparte na realnym popycie** - po teście kompetencji zawodowych model ML wskazuje konkretne kierunki kariery, koncentrując się na profesjach, które według prognoz będą szczególnie poszukiwane w ciągu najbliższych lat. Filtrowanie wg popytu jest opcjonalne.

## Technologie

- **FastAPI** - framework webowy
- **OpenAI** - embeddingi (`text-embedding-3-large`) i modele GPT-5
- **Supabase** - baza danych SQL
- **TensorFlow Lite** - sieć neuronowa do rekomendacji zawodów
- **NumPy** - obliczenia numeryczne

## Instalacja zależności

```bash
pip install -r requirements.txt
```

## Konfiguracja

Wymagane zmienne środowiskowe:

- `OPENAI_API_KEY` - klucz API OpenAI
- `SUPABASE_URL` - URL bazy Supabase
- `SUPABASE_SERVICE_ROLE_KEY` - klucz anon dla Supabase
- `OPENAI_RESPONSE_MODEL` - model OpenAI używany podczas czatowania (domyślnie: `gpt-5-mini`)
- `OPENAI_REASONING_EFFORT` - parametr oznaczający długość myślenia modelu OpenAI używanego podczas czatowania (do wyboru: `minimal`, `low`, `medium`, `high`, `veryHigh`; domyślnie: `low`)
- `OPENAI_CATEGORY_MODE` - model OpenAI do embeddingu, używany w starszym API opartego na kategoriach Porad (domyślnie: `text-embedding-3-large`) (DEPRECATED)
- `OPENAI_INTENT_MODEL` - model OpenAI do embeddingu, używany do porównywania znaczenia semantycznego wiadomości użytkownika na czacie i opisów porad w bazie danych
- `OPENAI_ADVICE_EMBEDDING_MODEL` - model OpenAI do embeddingu porad w bazie na podstawie ich opisów
- `ADVICE_SELECTION_MODE` - tryb wyboru porad: `categories` lub `embedding` (domyślnie: `embedding`, `categories` jest DEPRECATED)

## Uruchomienie

```bash
uvicorn app.main:app --reload --env-file .env
```

Lub używając skryptu:

```bash
./run.sh
```

## Endpointy HTTP

- `GET /advice` - rekomendacja porady psychologicznej
- `GET /career_adviser/advice` - rekomendacja porady zawodowej
- `POST /tests/psychology` - zapis wyników testu psychologicznego
- `GET /tests/psychology` - test psychologiczny użytkownika
- `POST /tests/vocation` - zapis wyników testu zawodowego
- `GET /tests/vocation` - test zawodowy użytkownika
- `GET /personas` - opis użytkownika wygenerowany przez LLM, używany przez LLM do dostosowania odpowiedzi na czacie
- `GET /health` - health check

Szczegóły nt. endpointu pod ścieżką /docs.
Pod ścieżką root znajduje się prosta strona .HTMl przeznaczona do lokalnych testów backendu (tryb czatu, test zawodowy, test psychologiczny), gdzie ręcznie wpisujemy ID użytkownika.

## Architektura aplikacji webowej

- **Repositories** - warstwa dostępu do danych (Supabase, in-memory)
- **Services** - logika aplikacyjna (selekcja porad, zamiana wyniku testu na wektor cech do sieci neuronowej)
- **Routers** - endpointy API
- **Models** - modele danych

Kod jest względnie modularny i zgodny z zasadą Dependency Inversion (DIP) - łatwo podmienić implementacje (np. inna baza danych, inne modele AI).

## Architektura sieci neuronowej
Sieć neuronowa na początku korzysta z dwóch warstw leakyReLU zapobiegającym obumieraniu neuronów. Trzecia warstwa to funkcja aktywacyjna sigmoid, zwracająca wynik dopasowania zawodu do użytkownika, w postaci procentowej. Użycie NumPy zamiast standardowych list skraca czas na odpowiedź.

## Uwagi

- Sieć neuronowa używa TensorFlow Lite dla oszczędności pamięci
- NumPy jest zablokowany na wersję <2.0 dla kompatybilności z tflite-runtime
- Dane popytu zawodów są ładowane z plików `data/inout/zawody.txt` i `data/inout/zawody5.txt`

## 🎯 Dalszy rozwój projektu
Pierwsza wersja backendu została przygotowana w dwa tygodnie, co wynika z ograniczeń czasu podczas hackathonu Hack Heroes. Z pewnością nie osiągnęliśmy takiego poziomu, który pozwala na "wypuszczenie w świat" naszego dzieła.

#### Główne problemy, z których zdajemy sobie sprawę:
- Ilość porad w naszej bazie danych jest mała (około 100).
- Algorytm wyboru porad nie jest precyzyjny, co w połączeniu z powyższym punktem sprawia, że jakość odpowiedzi nie jest najlepsza.
- Test zawodowy i test psychologiczny nie są w pełni skuteczne, a wiele pytań wymaga przeformułowania.
- Potrzebny jest lepszy algorytm zamieniający odpowiedzi w teście na cechy (*ang. features*) do sieci neuronowej.
- Dane do sieci neuronowej zostały, co prawda zachowując wszelkie środki ostrożności, wygenerowane przez algorytm. Zbiór nie jest zbyt obszerny (~7000 rekordów) i wymaga pozyskania realnych danych.

#### Plan dalszego rozwoju
- Rozważamy pójście w stronę celów społecznych, po dofinansowaniu od odpowiednich podmiotów, które pozwoliłyby ruszyć z projektem na większą skalę. Nie wykluczamy jednak skomercjalizowania projektu i stworzenia z niego usługi.
- Zamierzamy znacznie polepszyć jakość danych treningowych do modelu, a także ulepszyć jego architekturę.
- Potrzebny jest mechanizm feedbacku od użytkowników (kciuk w górę lub w dół), który pozwoli na udoskonalenie trybu zarówno Doradcy Życiowego, jak i Doradcy Zawodowego.
- Chcemy wypełnić "mockowe" ekrany prawdziwymi danymi nt. uczelni, kierunków i zawodów.
- Zamierzamy wprowadzić możliwość logowania się poprzez Google, a także swój adres mailowy.
- Będziemy musieli przemyśleć sposób działania aplikacji w chmurze, zabezpieczając się przed ewentualnymi atakami hakerskimi, być może zwiększyć możliwości serwera (aktualnie jest to pojedyncza maszyna Fly z 512MB RAM)
- Postaramy się wydać naszą aplikację na Google Play i App Store (po ewentualnym dopasowaniu aplikacji na system iOS)
- Chcemy, by AI znało kontekst całej rozmowy, jak w większości czatbotów. Póki co czatbot nie ma "pamięci". Będziemy nad tym pracować.
- Stawiamy duży nacisk na UX — chcemy, by aplikacja była lekka w obsłudze i wizualnie na poziomie współczesnych rozwiązań, z którymi młodzi (ale nie tylko młodzi) mają styczność na co dzień.