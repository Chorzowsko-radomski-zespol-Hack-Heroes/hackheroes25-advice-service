import numpy as np
from pathlib import Path

try:
    # Próbuj użyć tflite_runtime (lżejsza wersja, tylko do inferencji)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback do tensorflow.lite jeśli tflite_runtime nie jest dostępne
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

job = ["Strażak", "Policjant", "Ratownik medyczny", "Żołnierz", "Data scientist", "Data analyst", "Business analyst",
       "Specjalista ds cyberbezpieczeństwa", "Programista", "Statystyk", "Matematyk", "Fizyk", "Kierownik projektu",
       "Grafik komputerowy", "Projektant UI/UX", "Pracownik HR", "Konsultant IT", "Tester oprogramowania", "Trener personalny",
       "Ogrodnik", "Lekarz", "Weterynarz", "Farmaceuta", "Stomatolog", "Pielęgniarka", "Fizjoterapeuta", "Psycholog",
       "Psychiatra", "Coach", "Hydraulik", "Stolarz", "Mechanik", "Elektryk", "Kierownik magazynu", "Operator maszyn",
       "Projektant wnętrz", "Urbanista", "Kucharz", "Kelner", "Reżyser", "Aktor", "Muzyk", "Fotograf", "Pisarz", "Architekt",
       "Montażysta wideo", "Realizator dźwięku", "Kierownik sklepu", "Przedstawiciel handlowy", "Specjalista ds marketingu",
       "Specjalista ds social mediów", "Tłumacz", "Doradca_podatkowy", "Prawnik", "Dziennikarz", "Analityk finansowy",
       "Księgowy", "Agent ubezpieczeniowy", "Pracownik biura podróży", "Pilot samolotu", "Maszynista", "Kierowca", "Logistyk",
       "Nauczyciel", "Bibliotekarz", "Naukowiec", "Fryzjer", "Kosmetyczka", "Przedsiębiorca", "Komornik"]


def normalise(x, e=0.01):
    min_val = x.min()
    max_val = x.max()
    normali = (x-min_val)/(max_val-min_val)
    normali = normali*(1-2*e)+e
    return normali


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def _get_model_path() -> str:
    """Zwraca bezwzględną ścieżkę do modelu tflite."""
    base_path = Path(__file__).parent.parent
    model_path = base_path / "data" / "model.tflite"
    return str(model_path)


def _get_data_path(filename: str) -> str:
    """Zwraca bezwzględną ścieżkę do pliku danych."""
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data" / "inout" / filename
    return str(data_path)


def recommendations_tflite(personality_vector, wpep_mode, job_count):
    model_path = _get_model_path()
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array([personality_vector],
                          dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_indices = np.argsort(output_data)[-job_count:][::-1]
    recom = [(job[i], output_data[i]) for i in top_indices]
    trecom = transpose(recom)
    outscores = np.array(trecom[1])
    if wpep_mode == 0:
        nincome = np.ones(len(top_indices))
    else:
        if wpep_mode == 1:
            income = np.loadtxt(_get_data_path("wpep.csv"), delimiter=',')
        elif wpep_mode == 2:
            income = np.loadtxt(_get_data_path(
                "wpep5years.csv"), delimiter=',')
        else:
            # Fallback dla nieoczekiwanych wartości wpep_mode
            income = np.loadtxt(_get_data_path("wpep.csv"), delimiter=',')
        tincome = [income[i] for i in top_indices]
        nincome = normalise(np.array(tincome))
    out = outscores-(1-nincome)
    nout = normalise(out)
    return trecom[0], nout
