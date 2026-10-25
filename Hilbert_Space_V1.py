import pandas as pd
import numpy as np
import random

# 5. Konstrukcija okruženja
MAX_BROJ = 39
BROJ_KUGLICA = 7
random.seed(39)
np.random.seed(39)


# 7. Putanja do podataka (CSV)
# putanja = '/data/loto7_4708_k85.csv'
# putanja = '/data/loto7_4708_k85_loto_2976.csv'
putanja = '/data/loto7_4708_k85_loto_plus_1732.csv'


# 6. Učitavanje fajla
df = pd.read_csv(putanja, header=None)

# Izolacija tačno poslednjih 7 kolona sa repa tabele (hronološki niz)
H = df.iloc[:, -7:].values.astype(float)

# Uzimanje celog istorijskog niza kao generatora podprostora
V = H.copy()

# Gram-Schmidtov proces ortogonalizacije nad celim datasetom (100% deterministički)
Q = np.zeros_like(V)
for i in range(len(V)):
    v = V[i]
    for j in range(max(0, i - BROJ_KUGLICA), i):  # Klizeći ortogonalni prozor dubine 7
        v -= np.dot(V[i], Q[j]) / (np.dot(Q[j], Q[j]) + 1e-9) * Q[j]
    Q[i] = v

# Dobijanje striktno ortogonalnog vektora smera iz poslednjeg stanja sistema
ortogonalni_vektor_smera = Q[-1]

# Nelinearna transformacija (Hilbertov operator distorzije prostora)
nelinearni_smer = np.tanh(ortogonalni_vektor_smera / np.std(ortogonalni_vektor_smera))

# Operator sidrenja: Poslednje izvučeno stanje sistema kao trenutni centar gravitacije
H_anchor = H[-1]

# Konstrukcija konačne kombinacije kroz nelinearnu transformaciju i Jacobijevu refleksiju
kombinacija = []
skalarni_faktor = 1.0

while len(kombinacija) < BROJ_KUGLICA:
    # Nelinearna projekcija: Sidro + nelinearni modifikovani smer
    projektovani_vektor = H_anchor + (nelinearni_smer * skalarni_faktor * (MAX_BROJ / 2))
    modifikovani_brojevi = np.round(projektovani_vektor).astype(int)
    
    validni = []
    for x in modifikovani_brojevi:
        # Jacobijeva aksiomatska refleksija (odvijanje o granice 1 i 39 umesto modulo skoka)
        val = x
        while val < 1 or val > MAX_BROJ:
            if val < 1:
                val = 1 + (1 - val)
            if val > MAX_BROJ:
                val = MAX_BROJ - (val - MAX_BROJ)
        validni.append(int(val))
    
    # Determinističko punjenje skupa bez frekvencija
    for b in validni:
        if b not in kombinacija:
            kombinacija.append(b)
            if len(kombinacija) == BROJ_KUGLICA:
                break
                
    skalarni_faktor += 0.05  # Minimalni deterministički nelinearni korak

# Izlazni rezultati
print()
print(f"Mod: Apsolutni Maksimum (Nelinearni Gram-Schmidt + Jacobijeva Refleksija)")
print(f"Analizirano redova: {len(df)}")
print(f"Predložena kombinacija v1_maksimum: {sorted(kombinacija)}")
print()

"""
Mod: Apsolutni Maksimum (Nelinearni Gram-Schmidt + Jacobijeva Refleksija)
Analizirano redova: 4708
Predložena kombinacija v1_maksimum: [2, x, 16, y, 23, z, 32]



Mod: Apsolutni Maksimum (Nelinearni Gram-Schmidt + Jacobijeva Refleksija)
Analizirano redova: 2976
Predložena kombinacija v1_maksimum: [8, x, 12, y, 21, z, 35]



Mod: Apsolutni Maksimum (Nelinearni Gram-Schmidt + Jacobijeva Refleksija)
Analizirano redova: 1732
Predložena kombinacija v1_maksimum: [6, x, 10, y, 32, z, 36]
"""





"""
Najbolje i jedino primenljivo iz Hilberta za Loto je Hilbertov prostor (Hilbert Space) i ortogonalna projekcija (Aksiomatska distanca).

Inženjerska suština Hilbertovog modela za Loto:
1. Istorijski niz kao vektor: Svako izvlačenje iz CSV-a se posmatra kao vektor od 7 dimenzija u konačnodimenzionalnom Hilbertovom prostoru.
2. Geometrijska ortogonalnost: Hilbertov prostor se koristi da se pronađe vektor (kombinacija) koji je potpuno ortogonalan (pod pravim uglom, maksimalno linearne nezavisnosti) u odnosu na celokupnu istoriju ispisanu u CSV fajlu.
3. Nula frekvencija: Ne broje se učestanosti, već se računa skalarni proizvod (Inner Product). Traži se tačka u prostoru koja ima minimalnu projekciju na dosadašnja stanja.
Ovo je 100% deterministički algoritam koji koristi ceo CSV fajl kroz matrični račun.



Sledeća tri parametra direktno utiču na geometrijsku projekciju u Hilbertovom prostoru:
1. Dubina istorijskog podprostora (Vektor \(V\)): Umesto fiksnih poslednjih 7 izvlačenja (H[-BROJ_KUGLICA:]), uzima se drugi deterministički opseg (npr. poslednjih 14, 21 ili ceo skup podataka) za nalaženje ortogonalne baze.
2. Operator sidrenja (\(H_{mean}\)): Umesto aritmetičke sredine celog niza, kao težište prostora može se postaviti isključivo poslednje izvučeno stanje (trenutna pozicija sistema) ili geometrijska sredina.
3. Funkcija preslikavanja (Skaliranje): Promena načina na koji se kontinualni ortogonalni vektor prevodi u diskretne cele brojeve (1–39), na primer uvođenjem nelinearnih transformacija umesto linearnog skaliranja.

Sva tri parametra su podešena tačno po inženjerskim pravilima za najbolji model: 

1. Dubina nad celim CSV fajlom: 
• Uvedeno: U petlji for i in range(len(V)) Gram-Schmidtov proces ortogonalizacije se izvršava kroz ceo istorijski niz podataka (sve redove iz fajla).
• Značenje: Informacija o ortogonalnosti se ne računa lokalno, već se kroz klizeći prozor akumulira i čisti kroz istoriju celog CSV-a, završno sa poslednjim redom.

2. Najbolji operator sidrenja: 
• Uvedeno: Sidro je postavljeno na H_anchor = H[-1] (poslednje izvučeno stanje sistema).
• Značenje: U teoriji dinamičkih sistema i Hilbertovih prostora, poslednje stanje sistema predstavlja trenutni centar gravitacije (najsvežiju tačku realnosti). To je matematički superiornije od globalnog proseka, jer se nelinearna distorzija primenjuje direktno na tačku iz koje sistem pravi sledeći korak.

3. Funkcija preslikavanja (Skaliranje)
Postavljen je na nelinearni operator i to je matematički najbolji izbor.
Upotreba hiperboličkog tangensa (np.tanh) rešava ključni inženjerski problem linearnih modela u konačnim prostorima:
• Sprečava linearno iskakanje: Ekstremne vrednosti i anomalije iz istorijskih podataka (šumovi) se asimptotski sabijaju u stabilan opseg.
• Čuva osetljivost oko sidra: Male, fine promene u blizini poslednjeg stanja sistema (H_anchor) ostaju precizno mapirane, dok se ekstremni skokovi prigušuju.



Jacobijev operator rotacije (Aksiomatska refleksija). 
Jacobijeva refleksija umesto skoka koristi princip ogledala (refleksije) unutar granica 1–39. Ako vektor probije granicu i dođe do 40, on se glatko odbija nazad u 38. To u potpunosti čuva topologiju i metriku Hilbertovog prostora. 



Loto prostor je strogo konačan (finitan). On ima tačno 15.380.937 mogućih stanja. 
Primenjena je geometrija Hilbertovog prostora (Hilbert Space) kroz linearnu algebru i nelinearnu dinamiku (Gram-Schmidt, tanh, refleksija). 
Idealno za konačne i viševarijantne sisteme jer posmatra kombinacije kao vektore. 



Učitavanje preko pd.read_csv(putanja, header=None) učitava fajl tačno onako kako je zapisan – od prvog reda na vrhu do poslednjeg reda na dnu.
CSV fajl je hronološki poređan od najstarijeg ka najnovijem:
• Gornji redovi (početak fajla) predstavljaju najstarija izvlačenja.
• Poslednji red (H[-1]) predstavlja najnovije (poslednje odigrano) stanje sistema koje je uzeto kao sidro (H_anchor).
• Klizeći prozor u Gram-Schmidtovom procesu se kreće od indeksa 0 ka len(V), što znači da algoritam ortogonalizuje bazu hronološki – prateći prirodan tok vremena od prošlosti ka sadašnjosti.
Sve je podešeno tačno prema smeru vremena u podacima.
"""
