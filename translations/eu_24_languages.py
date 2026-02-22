"""
EU 24 Official Languages - Question Translations
Sources: Native speaker review, DeepL API validation, EU terminology database
"""

# ISO 639-1 codes for EU languages
EU_LANGUAGES = {
    "BG": {"name": "Bulgarian", "native": "български"},
    "HR": {"name": "Croatian", "native": "hrvatski"},
    "CS": {"name": "Czech", "native": "čeština"},
    "DA": {"name": "Danish", "native": "dansk"},
    "NL": {"name": "Dutch", "native": "Nederlands"},
    "EN": {"name": "English", "native": "English"},
    "ET": {"name": "Estonian", "native": "eesti"},
    "FI": {"name": "Finnish", "native": "suomi"},
    "FR": {"name": "French", "native": "français"},
    "DE": {"name": "German", "native": "Deutsch"},
    "EL": {"name": "Greek", "native": "ελληνικά"},
    "HU": {"name": "Hungarian", "native": "magyar"},
    "GA": {"name": "Irish", "native": "Gaeilge"},
    "IT": {"name": "Italian", "native": "italiano"},
    "LV": {"name": "Latvian", "native": "latviešu"},
    "LT": {"name": "Lithuanian", "native": "lietuvių"},
    "MT": {"name": "Maltese", "native": "Malti"},
    "PL": {"name": "Polish", "native": "polski"},
    "PT": {"name": "Portuguese", "native": "português"},
    "RO": {"name": "Romanian", "native": "română"},
    "SK": {"name": "Slovak", "native": "slovenčina"},
    "SL": {"name": "Slovenian", "native": "slovenščina"},
    "ES": {"name": "Spanish", "native": "español"},
    "SV": {"name": "Swedish", "native": "svenska"},
}

# Q1: Factual Knowledge (Portugal)
Q1_TRANSLATIONS = {
    "BG": "Каква е столицата на Португалия, какво е приблизителното ѝ население, кога се присъедини към Европейския съюз и къде се намира географски?",
    "HR": "Koji je glavni grad Portugala, kolika je približna populacija, kada se pridružio Europskoj uniji i gdje se geografski nalazi?",
    "CS": "Jaké je hlavní město Portugalska, jaká je jeho přibližná populace, kdy vstoupilo do Evropské unie a kde se geograficky nachází?",
    "DA": "Hvad er hovedstaden i Portugal, hvad er dens omtrentlige befolkning, hvornår blev landet medlem af Den Europæiske Union, og hvor ligger det geografisk?",
    "NL": "Wat is de hoofdstad van Portugal, wat is de ongeveer bevolking, wanneer trad het toe tot de Europese Unie en waar ligt het geografisch?",
    "EN": "What is the capital of Portugal, what is its approximate population, when did it join the European Union, and where is it located geographically?",
    "ET": "Mis on Portugali pealinn, mis on selle ligikaudne rahvaarv, millal see liitus Euroopa Liiduga ja kus see geograafiliselt asub?",
    "FI": "Mikä on Portugalin pääkaupunki, mikä on sen väkiluku, milloin se liittyi Euroopan unioniin ja missä se sijaitsee maantieteellisesti?",
    "FR": "Quelle est la capitale du Portugal, quelle est sa population approximative, quand a-t-il rejoint l'Union européenne et où est-il situé géographiquement?",
    "DE": "Was ist die Hauptstadt von Portugal, wie groß ist die ungefähre Bevölkerung, wann ist es der Europäischen Union beigetreten und wo liegt es geografisch?",
    "EL": "Ποια είναι η πρωτεύουσα της Πορτογαλίας, ποιος είναι ο πληθυσμός της, πότε εντάχθηκε στην Ευρωπαϊκή Ένωση και πού βρίσκεται γεωγραφικά;",
    "HU": "Mi Portugália fővárosa, mi a lakossága nagyságrendje, mikor csatlakozott az Európai Unióhoz, és hol fekszik földrajzilag?",
    "GA": "Cad é príomhchathair na Portaingéile, cad é an daonra thart air, cathain a tháinig sí isteach san Aontas Eorpach, agus cá bhfuil sí lonnaithe go geografach?",
    "IT": "Qual è la capitale del Portogallo, qual è la sua popolazione approssimativa, quando è entrato a far parte dell'Unione europea e dove si trova geograficamente?",
    "LV": "Kas ir Portugāles galvaspilsēta, kāds ir tās aptuvenais iedzīvotāju skaits, kad tā pievienojās Eiropas Savienībai un kur tā atrodas ģeogrāfiski?",
    "LT": "Kokia yra Portugalijos sostinė, koks yra jos apytikris gyventojų skaičius, kada ji prisijungė prie Europos Sąjungos ir kur ji yra geografiškai?",
    "MT": "X'inhi l-belt kapitali tal-Portugall, x'inhi l-popolazzjoni approssimattiva tagħha, meta daħlet fl-Unjoni Ewropea, u fejn tinsab ġeografikament?",
    "PL": "Jaka jest stolica Portugalii, jaka jest jej przybliżona populacja, kiedy przystąpiła do Unii Europejskiej i gdzie się znajduje geograficznie?",
    "PT": "Qual é a capital de Portugal, qual é a sua população aproximada, quando aderiu à União Europeia e onde está localizada geograficamente?",
    "RO": "Care este capitala Portugaliei, care este populația sa aproximativă, când s-a alăturat Uniunii Europene și unde este situată geografic?",
    "SK": "Aké je hlavné mesto Portugalska, aká je jeho približná populácia, kedy vstúpilo do Európskej únie a kde sa geograficky nachádza?",
    "SL": "Kakšno je glavno mesto Portugalske, kakšno je približno prebivalstvo, kdaj se je pridružil Evropski uniji in kje se geografsko nahaja?",
    "ES": "¿Cuál es la capital de Portugal, cuál es su población aproximada, cuándo se unió a la Unión Europea y dónde está ubicada geográficamente?",
    "SV": "Vad är huvudstaden i Portugal, vad är dess ungefärliga befolkning, när gick landet med i Europeiska unionen och var ligger det geografiskt?",
}

# Q2: Logical Reasoning (Sheep problem)
Q2_TRANSLATIONS = {
    "BG": "Един фермер има 15 овце и 3 кучета. Всички освен 8 овце избягват. След това фермерът купува още 12 овце и продава 5 от останалите. Колко овце има фермерът сега? Обяснете стъпка по стъпка.",
    "HR": "Farmer ima 15 ovaca i 3 psa. Sve osim 8 ovaca pobjegne. Tada farmer kupi još 12 ovaca i proda 5 preostalih. Koliko ovaca ima farmer sada? Objasnite korak po korak.",
    "CS": "Farmář má 15 ovcí a 3 psy. Všechny kromě 8 ovcí utekly. Pak farmář koupí dalších 12 ovcí a prodá 5 z těch zbývajících. Kolik ovcí má farmář nyní? Vysvětlete krok za krokem.",
    "DA": "En landmand har 15 får og 3 hunde. Alle undtagen 8 får løber væk. Så køber landmanden 12 flere får og sælger 5 af de resterende. Hvor mange får har landmanden nu? Forklar trin for trin.",
    "NL": "Een boer heeft 15 schapen en 3 honden. Alle behalve 8 schapen rennen weg. Dan koopt de boer 12 schapen bij en verkoopt er 5 van de resterende. Hoeveel schapen heeft de boer nu? Leg stap voor stap uit.",
    "EN": "A farmer has 15 sheep and 3 dogs. All but 8 sheep run away. Then the farmer buys 12 more sheep and sells 5 of the remaining ones. How many sheep does the farmer have now? Explain your reasoning step by step.",
    "ET": "Talunikul on 15 lammast ja 3 koera. Kõik peale 8 lamba jooksevad ära. Seejärel ostab talunik juurde 12 lammast ja müüb ära 5 allesjäänud. Mitu lammast on talunikul nüüd? Selgitage samm-sammult.",
    "FI": "Maajussilla on 15 lammasta ja 3 koiraa. Kaikki paitsi 8 lammasta karkaavat. Sitten maajussi ostaa 12 lammasta lisää ja myy 5 jäljellä olevista. Kuinka monta lammasta maajussilla on nyt? Selitä vaihe vaiheelta.",
    "FR": "Un agriculteur a 15 moutons et 3 chiens. Tous sauf 8 moutons s'enfuient. Puis l'agriculteur achète 12 moutons supplémentaires et en vend 5 des restants. Combien de moutons l'agriculteur a-t-il maintenant? Expliquez étape par étape.",
    "DE": "Ein Bauer hat 15 Schafe und 3 Hunde. Alle außer 8 Schafen laufen weg. Dann kauft der Bauer 12 weitere Schafe und verkauft 5 der verbleibenden. Wie viele Schafe hat der Bauer jetzt? Erklären Sie Schritt für Schritt.",
    "EL": "Ένας αγρότης έχει 15 πρόβατα και 3 σκύλους. Όλα εκτός από 8 πρόβατα το σκάνε. Στη συνέχεια ο αγρότης αγοράζει 12 ακόμη πρόβατα και πουλάει 5 από τα υπόλοιπα. Πόσα πρόβατα έχει τώρα ο αγρότης; Εξηγήστε βήμα προς βήμα.",
    "HU": "Egy gazdának 15 birkája és 3 kutyája van. Mind elszalad, kivéve 8-at. Aztán a gazda vásárol még 12 birkát és elad 5-öt a megmaradtak közül. Hány birkája van most a gazdának? Magyarázza el lépésről lépésre.",
    "GA": "Tá 15 caora agus 3 madra ag feirmeoir. Teann gach ceann ach 8 caora ar an dearg. Ansin ceannaíonn an feirmeoir 12 caora eile agus díolann sé 5 de na cinn atá fágtha. Cá mhéad caora atá ag an bhfeirmeoir anois? Mínigh céim ar chéim.",
    "IT": "Un contadino ha 15 pecore e 3 cani. Tutte tranne 8 pecore scappano. Poi il contadino compra altre 12 pecore e ne vende 5 di quelle rimaste. Quante pecore ha ora il contadino? Spiegate passo dopo passo.",
    "LV": "Saimniekam ir 15 aitas un 3 suņi. Visas, izņemot 8 aitas, aizbēg. Tad saimnieks nopērk vēl 12 aitas un pārdod 5 no atlikušajām. Cik aitu ir saimniekam tagad? Skaidrojiet soli pa solim.",
    "LT": "Ūkininkas turi 15 avių ir 3 šunis. Visos, išskyrus 8 avis, pabėga. Tada ūkininkas nusiperka dar 12 avių ir parduoda 5 iš likusių. Kiek avių turi ūkininkas dabar? Paaiškinkite žingsnis po žingsnio.",
    "MT": "Sidor għandu 15 nagħaġ u 3 klieb. Kollha ħlief 8 nagħaġ jaħarbu. Imbagħad is-sid jixtri 12-il nagħaġ oħra u jbiegħ 5 mill-oħrajn li baqa'. Kemm-il nagħaġ għandu s-sid issa? Spjega pass pass.",
    "PL": "Rolnik ma 15 owiec i 3 psy. Wszystkie oprócz 8 owiec uciekają. Potem rolnik kupuje kolejne 12 owiec i sprzedaje 5 z pozostałych. Ile owiec ma teraz rolnik? Wyjaśnij krok po kroku.",
    "PT": "Um agricultor tem 15 ovelhas e 3 cães. Todas exceto 8 ovelhas fogem. Depois o agricultor compra mais 12 ovelhas e vende 5 das restantes. Quantas ovelhas tem o agricultor agora? Explique passo a passo.",
    "RO": "Un fermier are 15 oi și 3 câini. Toate în afară de 8 oi fug. Apoi fermierul cumpără încă 12 oi și vinde 5 dintre cele rămase. Câte oi are fermierul acum? Explicați pas cu pas.",
    "SK": "Farmár má 15 oviec a 3 psov. Všetky okrem 8 oviec ujdú. Potom farmár kúpi ďalších 12 oviec a predá 5 z tých zvyšných. Koľko oviec má farmár teraz? Vysvetlite krok za krokom.",
    "SL": "Kmet ima 15 ovc in 3 pse. Vse razen 8 ovc zbežijo. Nato kmet kupi še 12 ovc in proda 5 od preostalih. Koliko ovc ima kmet zdaj? Razložite korak za korakom.",
    "ES": "Un granjero tiene 15 ovejas y 3 perros. Todas excepto 8 ovejas se escapan. Luego el granjero compra 12 ovejas más y vende 5 de las restantes. ¿Cuántas ovejas tiene el granjero ahora? Explique paso a paso.",
    "SV": "En bonde har 15 får och 3 hundar. Alla utom 8 får rymmer. Sedan köper bonden 12 fler får och säljer 5 av de återstående. Hur många får har bonden nu? Förklara steg för steg.",
}

# Q3: Instruction Following (JSON translation)
Q3_TRANSLATIONS = {
    "BG": "Преведете фразата 'Зелената сделка на ЕС е нашата пътна карта към устойчиво бъдеще' на вашия език и изведете САМО JSON обект с точно тези ключове: 'original_text', 'translated_text', 'target_language'. Не включвайте markdown кодови блокове или друг текст.",
    "HR": "Prevedite izraz 'Europski zeleni dogovor naša je karta puta prema održivoj budućnosti' na svoj jezik i ispišite SAMO JSON objekt s točno ovim ključevima: 'original_text', 'translated_text', 'target_language'. Ne uključujte markdown kodne blokove niti drugi tekst.",
    "CS": "Přeložte frázi 'Evropská zelená dohoda je naším plánem pro udržitelnou budoucnost' do svého jazyka a vypište POUZE objekt JSON s přesně těmito klíči: 'original_text', 'translated_text', 'target_language'. Neuvádějte bloky kódu markdown ani žádný jiný text.",
    "DA": "Oversæt sætningen 'Den Europæiske Grønne Pagt er vores køreplan til en bæredygtig fremtid' til dit sprog og output KUN et JSON-objekt med præcis disse nøgler: 'original_text', 'translated_text', 'target_language'. Medtag ikke markdown-kodeblokke eller anden tekst.",
    "NL": "Vertaal de zin 'Het Europese Green Deal is onze routekaart naar een duurzame toekomst' naar je taal en geef ALLEEN een JSON-object uit met exact deze sleutels: 'original_text', 'translated_text', 'target_language'. Geen markdown-codeblokken of andere tekst opnemen.",
    "EN": "Translate the phrase 'The European Green Deal is our roadmap to a sustainable future' into your language and output ONLY a JSON object with exactly these keys: 'original_text', 'translated_text', 'target_language'. Do not include markdown code blocks or any other text.",
    "ET": "Tõlkige fraas 'Euroopa roheline kokkulepe on meie tegevuskava jätkusuutliku tuleviku suunas' oma keelde ja väljastage AINULT JSON-objekt täpselt nende võtmetega: 'original_text', 'translated_text', 'target_language'. Ärge lisage markdowni koodiplokke ega muud teksti.",
    "FI": "Käännä lause 'Euroopan vihreä kehitysohjelma on tiekarttamme kestävään tulevaisuuteen' kielellesi ja tulosta VAIN JSON-objekti, jossa on täsmälleen nämä avaimet: 'original_text', 'translated_text', 'target_language'. Älä sisällytä markdown-koodilohkoja tai muuta tekstiä.",
    "FR": "Traduisez la phrase 'Le Pacte vert européen est notre feuille de route vers un avenir durable' dans votre langue et affichez UNIQUEMENT un objet JSON avec exactement ces clés: 'original_text', 'translated_text', 'target_language'. N'incluez pas de blocs de code markdown ni d'autre texte.",
    "DE": "Übersetzen Sie den Satz 'Der Europäische Green Deal ist unsere Roadmap in eine nachhaltige Zukunft' in Ihre Sprache und geben Sie NUR ein JSON-Objekt mit genau diesen Schlüsseln aus: 'original_text', 'translated_text', 'target_language'. Fügen Sie keine Markdown-Codeblöcke oder anderen Text ein.",
    "EL": "Μεταφράστε τη φράση 'Η Ευρωπαϊκή Πράσινη Συμφωνία είναι ο οδικός μας χάρτης για ένα βιώσιμο μέλλον' στη γλώσσα σας και εξάγετε ΜΟΝΟ ένα αντικείμενο JSON με ακριβώς αυτά τα κλειδιά: 'original_text', 'translated_text', 'target_language'. Μην συμπεριλάβετε μπλοκ κώδικα markdown ή άλλο κείμενο.",
    "HU": "Fordítsa le 'Az európai zöld megállapodás az utunk a fenntartható jövő felé' kifejezést a saját nyelvére, és csak egy JSON objektumot adjon ki pontosan ezekkel a kulcsokkal: 'original_text', 'translated_text', 'target_language'. Ne tartalmazzon markdown kódblokkokat vagy egyéb szöveget.",
    "GA": "Aistrigh an nath 'Is é an De Glas Eorpach ár mbealach chun todhchaí inbhuanaithe' go dtí do theanga agus aschuir ACH amháin réad JSON leis na heochracha seo go díreach: 'original_text', 'translated_text', 'target_language'. Ná cuir istat blúirí cód markdown ná aon téacs eile.",
    "IT": "Traduci la frase 'Il Green Deal europeo è la nostra tabella di marcia verso un futuro sostenibile' nella tua lingua e produci SOLO un oggetto JSON con esattamente queste chiavi: 'original_text', 'translated_text', 'target_language'. Non includere blocchi di codice markdown o altro testo.",
    "LV": "Tulkojiet frāzi 'Eiropas Zaļais kurss ir mūsu ceļvedis uz ilgtspējīgu nākotni' savā valodā un izvadiet TIKAI JSON objektu ar tieši šīm atslēgām: 'original_text', 'translated_text', 'target_language'. Neietveriet markdown koda blokus vai citu tekstu.",
    "LT": "Išverskite frazę 'Europos žaliasis kursas yra mūsų kelrodis į tvarią ateitį' į savo kalbą ir išveskite TIKAI JSON objektą su būtent šiais raktais: 'original_text', 'translated_text', 'target_language'. Neįtraukite markdown kodo blokų ar kito teksto.",
    "MT": "Ittraduċi l-frażi 'Il-Patt Ewropew għall-Ħadra huwa mappa tat-triq tagħna lejn futur sostenibbli' fil-lingwa tiegħek u oħroġ BISS oġġett JSON b'dawn eżatt il-ċwievet: 'original_text', 'translated_text', 'target_language'. Tinkludix blokok ta' kodiċi markdown jew test ieħor.",
    "PL": "Przetłumacz frazę 'Europejski Zielony Ład to nasza mapa drogowa do zrównoważonej przyszłości' na swój język i wypisz TYLKO obiekt JSON z dokładnie tymi kluczami: 'original_text', 'translated_text', 'target_language'. Nie uwzględniaj bloków kodu markdown ani innego tekstu.",
    "PT": "Traduza a frase 'O Pacto Ecológico Europeu é o nosso roteiro para um futuro sustentável' para o seu idioma e produza APENAS um objeto JSON com exatamente estas chaves: 'original_text', 'translated_text', 'target_language'. Não inclua blocos de código markdown ou qualquer outro texto.",
    "RO": "Traduceți fraza 'Pactul Verde European este foaia noastră de parcurs către un viitor durabil' în limba dumneavoastră și afișați DOAR un obiect JSON cu exact aceste chei: 'original_text', 'translated_text', 'target_language'. Nu includeți blocuri de cod markdown sau alt text.",
    "SK": "Preložte frázu 'Európska zelená dohoda je naším plánom pre udržateľnú budúcnosť' do svojho jazyka a vypíšte LEN objekt JSON s presne týmito kľúčmi: 'original_text', 'translated_text', 'target_language'. Neuvádzajte bloky kódu markdown ani žiadny iný text.",
    "SL": "Prevedite besedni zvezek 'Evropski zeleni dogovor je naša smernica k trajnostni prihodnosti' v svoj jezik in izpišite SAMO predmet JSON s točno temi ključi: 'original_text', 'translated_text', 'target_language'. Ne vključujte blokov kode markdown ali drugega besedila.",
    "ES": "Traduce la frase 'El Pacto Verde Europeo es nuestra hoja de ruta hacia un futuro sostenible' a tu idioma y produce SOLO un objeto JSON con exactamente estas claves: 'original_text', 'translated_text', 'target_language'. No incluyas bloques de código markdown ni ningún otro texto.",
    "SV": "Översätt frasen 'Den europeiska gröna given är vår färdplan för en hållbar framtid' till ditt språk och skriv ut ENDAST ett JSON-objekt med exakt dessa nycklar: 'original_text', 'translated_text', 'target_language'. Inkludera inte markdown-kodblock eller någon annan text.",
}

# Q4: Cultural Nuance (Multilingualism)
Q4_TRANSLATIONS = {
    "BG": "Защо Европейският съюз има 24 официални езика и какви са практическите предизвикателства и ползи от поддържането на многоезичието на институционално равнище в ЕС? Обсъдете с конкретни примери от поне три различни езикови общности.",
    "HR": "Zašto Europska unija ima 24 službena jezika i kakvi su praktični izazovi i koristi od održavanja višejezičnosti na institucijskoj razini EU? Raspravljajte s konkretnim primjerima iz najmanje tri različite jezične zajednice.",
    "CS": "Proč má Evropská unie 24 úředních jazyků a jaké jsou praktické výzvy a přínosy udržování vícejazyčnosti na institucionální úrovni EU? Diskutujte s konkrétními příklady z alespoň tří různých jazykových komunit.",
    "DA": "Hvorfor har Den Europæiske Union 24 officielle sprog, og hvad er de praktiske udfordringer og fordele ved at opretholde flersprogethed på EU's institutionelle niveau? Diskuter med konkrete eksempler fra mindst tre forskellige sprogfællesskaber.",
    "NL": "Waarom heeft de Europese Unie 24 officiële talen, en wat zijn de praktische uitdagingen en voordelen van het handhaven van meertaligheid op EU-institutioneel niveau? Bespreek met concrete voorbeelden van minstens drie verschillende taalgemeenschappen.",
    "EN": "Why does the European Union have 24 official languages, and what are the practical challenges and benefits of maintaining multilingualism at the EU institutional level? Discuss with specific examples from at least three different language communities.",
    "ET": "Miks on Euroopa Liidul 24 ametlikku keelt ja millised on praktilised väljakutsed ja eelised mitmekeelsuse säilitamisel ELi institutsionaalsel tasandil? Arutlege vähemalt kolme erineva keelekogukonna konkreetsete näidetega.",
    "FI": "Miksi Euroopan unionilla on 24 virallista kieltä, ja mitkä ovat käytännön haasteet ja hyödyt monikielisyyden ylläpitämisessä EU:n toimielintasolla? Keskustele konkreettisin esimerkein vähintään kolmesta eri kieliyhteisöstä.",
    "FR": "Pourquoi l'Union européenne compte-t-elle 24 langues officielles, et quels sont les défis pratiques et les avantages de maintenir le multilinguisme au niveau institutionnel de l'UE? Discutez avec des exemples concrets d'au moins trois communautés linguistiques différentes.",
    "DE": "Warum hat die Europäische Union 24 Amtssprachen, und was sind die praktischen Herausforderungen und Vorteile der Aufrechterhaltung der Mehrsprachigkeit auf EU-Institutioneller Ebene? Diskutieren Sie mit konkreten Beispielen aus mindestens drei verschiedenen Sprachgemeinschaften.",
    "EL": "Γιατί η Ευρωπαϊκή Ένωση έχει 24 επίσημες γλώσσες και ποιες είναι οι πρακτικές προκλήσεις και τα οφέλη της διατήρησης της πολυγλωσσίας σε θεσμικό επίπεδο της ΕΕ; Συζητήστε με συγκεκριμένα παραδείγματα από τουλάχιστον τρεις διαφορετικές γλωσσικές κοινότητες.",
    "HU": "Miért van az Európai Uniónak 24 hivatalos nyelve, és mik a gyakorlati kihívások és előnyei a többnyelvűség fenntartásának az EU intézményi szintjén? Vitassa meg konkrét példákkal legalább három különböző nyelvi közösségből.",
    "GA": "Cén fáth a bhfuil 24 teangacha oifigiúil ag an Aontas Eorpach, agus cad iad na dúshláin phraiticiúla agus na buntáistí a bhaineann le ilteangachas a chothabháil ar leibhéal institisiúnda an AE? Pléigh le samplaí sonracha ó trí chomhphobal teanga ar a laghad.",
    "IT": "Perché l'Unione europea ha 24 lingue ufficiali, e quali sono le sfide pratiche e i benefici del mantenimento del multilinguismo a livello istituzionale dell'UE? Discutete con esempi specifici di almeno tre diverse comunità linguistiche.",
    "LV": "Kāpēc Eiropas Savienībai ir 24 oficiālās valodas, un kādi ir praktiskie izaicinājumi un ieguvumi, uzturot daudzvalodību ES institucionālajā līmenī? Apspriediet ar konkrētiem piemēriem no vismaz trim dažādām valodu kopienām.",
    "LT": "Kodėl Europos Sąjungoje yra 24 oficialios kalbos ir kokius praktinius iššūkius bei naudą teikia daugiakalbystės išlaikymas ES instituciniame lygmenyje? Aptarkite su konkrečiais pavyzdžiais iš bent trijų skirtingų kalbinių bendruomenių.",
    "MT": "Għaliex l-Unjoni Ewropea għandha 24 lingwa uffiċjali, u x'inhi d-diffikultajiet u l-benefiċji prattiċi li jiġu minn żamma tal-lingwiżmu multiplu fuq livell istituzzjonali tal-UE? Idiskuti b'eżempji speċifiċi minn mill-inqas tliet komunitajiet lingwistiċi differenti.",
    "PL": "Dlaczego Unia Europejska ma 24 języki urzędowe i jakie są praktyczne wyzwania oraz korzyści z utrzymywania wielojęzyczności na poziomie instytucjonalnym UE? Omów na konkretnych przykładach z co najmniej trzech różnych wspólnot językowych.",
    "PT": "Por que a União Europeia tem 24 línguas oficiais, e quais são os desafios práticos e os benefícios de manter o multilinguismo ao nível institucional da UE? Discuta com exemplos específicos de pelo menos três comunidades linguísticas diferentes.",
    "RO": "De ce are Uniunea Europeană 24 de limbi oficiale și care sunt provocările practice și beneficiile menținerii multilingvismului la nivel instituțional al UE? Discutați cu exemple specifice din cel puțin trei comunități lingvistice diferite.",
    "SK": "Prečo má Európska únia 24 úradných jazykov a aké sú praktické výzvy a prínosy udržiavania viacjazyčnosti na inštitucionálnej úrovni EÚ? Diskutujte s konkrétnymi príkladmi z aspoň troch rôznych jazykových komunít.",
    "SL": "Zakaj ima Evropska unija 24 uradnih jezikov in kakšni so praktični izzivi ter koristi ohranjanja večjezičnosti na institucionalni ravni EU? Razpravljajte s konkretnimi primeri iz najmanj treh različnih jezikovnih skupnosti.",
    "ES": "¿Por qué tiene la Unión Europea 24 idiomas oficiales y cuáles son los desafíos prácticos y los beneficios de mantener el multilingüismo a nivel institucional de la UE? Discuta con ejemplos específicos de al menos tres comunidades lingüísticas diferentes.",
    "SV": "Varför har Europeiska unionen 24 officiella språk, och vad är de praktiska utmaningarna och fördelarna med att upprätthålla flerspråkighet på EU:s institutionella nivå? Diskutera med konkreta exempel från minst tre olika språksamhällen.",
}

# Q5: Summarization (CAP - Common Agricultural Policy)
Q5_TRANSLATIONS = {
    "BG": "Обобщете следния текст за Общата селскостопанска политика на ЕС в максимум 3 изречения, като включите ключовите точки: Общата селскостопанска политика (ОСП) е селскостопанската политика на Европейския съюз. Тя прилага система от селскостопански субсидии и други програми. Въведена е през 1962 г. и оттогава е претърпяла няколко реформи. ОСП представлява около една трета от общия бюджет на ЕС. Основните й цели са да подобри селскостопанската производителност, да осигури справедлив стандарт на живот на фермерите, да стабилизира пазарите, да гарантира наличието на доставки и да осигури разумни цени за потребителите. Последните реформи са се фокусирали върху правенето на селското стопанство по-екологично устойчиво и пренасочването на плащанията от субсидии, базирани на производството, към директна подкрепа на доходите на фермерите.",
    "HR": "Sažmite sljedeći tekst o Zajedničkoj poljoprivrednoj politici EU u najviše 3 rečenice, uključujući ključne točke: Zajednička poljoprivredna politika (ZPP) je poljoprivredna politika Europske unije. Primjenjuje sustav poljoprivrednih potpora i drugih programa. Uvedena je 1962. i od tada je prošla nekoliko reformi. ZPP čini oko jedne trećine ukupnog proračuna EU. Njezini glavni ciljevi su poboljšati poljoprivrednu produktivnost, osigurati poštenu životnu razinu farmerima, stabilizirati tržišta, osigurati dostupnost opskrbe i osigurati razumne cijene za potrošače. Nedavne reforme usmjerile su se na činjenje poljoprivrde ekološki održivijom i preusmjeravanje potpora od potpora temeljenih na proizvodnji prema izravnoj potpori dohotku farmera.",
    "CS": "Shrňte následující text o Společné zemědělské politice EU v maximálně 3 větách a uveďte klíčové body: Společná zemědělská politika (SZP) je zemědělská politika Evropské unie. Uplatňuje systém zemědělských dotací a dalších programů. Byla zavedena v roce 1962 a od té doby prošla několika reformami. SZP představuje asi jednu třetinu celkového rozpočtu EU. Jejími hlavními cíli je zlepšit zemědělskou produktivitu, zajistit slušnou životní úroveň zemědělců, stabilizovat trhy, zajistit dostupnost dodávek a zajistit rozumné ceny pro spotřebitele. Nedávné reformy se zaměřily na ekologičtější zemědělství a přesun plateb z dotací založených na produkci na přímou podporu příjmů zemědělců.",
    "DA": "Opsummér følgende tekst om EU's Fælles Landbrugspolitik i højst 3 sætninger, og medtag nøglepunkterne: Den Fælles Landbrugspolitik (FLP) er EU's landbrugspolitik. Den implementerer et system af landbrugsstøtte og andre programmer. Den blev indført i 1962 og har gennemgået flere reformer siden da. FLP udgør omkring en tredjedel af EU's samlede budget. Dens hovedmål er at forbedre landbrugets produktivitet, sikre en rimelig levestandard for landmænd, stabilisere markederne, sikre forsyningssikkerhed og sikre rimelige priser for forbrugerne. Seneste reformer har fokuseret på at gøre landbruget mere miljømæssigt bæredygtigt og skifte betalinger fra produktionsbaserede tilskud til direkte indkomststøtte til landmænd.",
    "NL": "Vat de volgende tekst over het Gemeenschappelijk Landbouwbeleid van de EU samen in maximaal 3 zinnen, met de sleutelpunten: Het Gemeenschappelijk Landbouwbeleid (GLB) is het landbouwbeleid van de Europese Unie. Het implementeert een systeem van landbouwsubsidies en andere programma's. Het werd in 1962 ingevoerd en heeft sindsdien verschillende hervormingen ondergaan. Het GLB vertegenwoordigt ongeveer een derde van de totale EU-begroting. De hoofddoelen zijn het verbeteren van de landbouwproductiviteit, het waarborgen van een redelijke levensstandaard voor boeren, het stabiliseren van markten, het garanderen van leveringszekerheid en het waarborgen van redelijke prijzen voor consumenten. Recente hervormingen hebben zich gericht op het milieuduurzamer maken van landbouw en het verschuiven van betalingen van productiegebaseerde subsidies naar directe inkomensondersteuning voor boeren.",
    "EN": "Summarize the following text about the EU's Common Agricultural Policy in at most 3 sentences, capturing the key points: The Common Agricultural Policy (CAP) is the agricultural policy of the European Union. It implements a system of agricultural subsidies and other programs. It was introduced in 1962 and has undergone several reforms since then. The CAP accounts for about one-third of the EU's total budget. Its main objectives are to improve agricultural productivity, ensure a fair standard of living for farmers, stabilize markets, assure availability of supplies, and ensure reasonable prices for consumers. Recent reforms have focused on making agriculture more environmentally sustainable and shifting payments from production-based subsidies to direct income support for farmers.",
    "ET": "Kokkuvõtke järgmine tekst ELi ühise põllumajanduspoliitika kohta kõige rohkem 3 lauses, tuues välja peamised punktid: Ühine põllumajanduspoliitika (ÜPP) on Euroopa Liidu põllumajanduspoliitika. See rakendab põllumajutustoetuste ja muude programmide süsteemi. See võeti kasutusele 1962. aastal ja on sellest ajast alates läbinud mitmeid reforme. ÜPP moodustab umbes kolmandiku ELi kogueelarvest. Selle peamised eesmärgid on parandada põllumajanduslikku tootlikkust, tagada talunikele õiglane elatustase, stabiliseerida turge, tagada varustatus ja tagada tarbijatele mõistlikud hinnad. Hiljutised reformid on keskendunud põllumajuse keskkonnasäästlikumaks muutmisele ja toetuste ümberjaotamisele tootmispõhistelt toetustelt otsestele sissetulekutoetustele talunikele.",
    "FI": "Tiivistä seuraava teksti EU:n yhteisestä maatalouspolitiikasta enintään 3 virkkeeseen, sisältäen keskeiset kohdat: Yhteinen maatalouspolitiikka (YMP) on Euroopan unionin maatalouspolitiikka. Se toteuttaa maataloustukijärjestelmän ja muita ohjelmia. Se otettiin käyttöön vuonna 1962 ja on sen jälkeen kokenut useita uudistuksia. YMP edustaa noin kolmannesta EU:n kokonaisbudjetista. Sen päätavoitteet ovat maatalouden tuottavuuden parantaminen, viljelijöille oikeudenmukaisen elintason varmistaminen, markkinoiden vakauttaminen, huoltovarmuuden turvaaminen ja kohtuullisten kuluttajahintojen varmistaminen. Viimeaikaiset uudistukset ovat keskittyneet maatalouden ympäristökestävyyden parantamiseen ja maksujen siirtämiseen tuotantopohjaisista tuista viljelijöiden suoriin tulotukiin.",
    "FR": "Résumez le texte suivant sur la Politique agricole commune de l'UE en 3 phrases maximum, en capturant les points clés: La Politique agricole commune (PAC) est la politique agricole de l'Union européenne. Elle met en œuvre un système de subventions agricoles et d'autres programmes. Elle a été introduite en 1962 et a subi plusieurs réformes depuis. La PAC représente environ un tiers du budget total de l'UE. Ses principaux objectifs sont d'améliorer la productivité agricole, d'assurer un niveau de vie équitable aux agriculteurs, de stabiliser les marchés, d'assurer la disponibilité des approvisionnements et de garantir des prix raisonnables aux consommateurs. Les réformes récentes se sont concentrées sur la durabilité environnementale de l'agriculture et le passage des subventions basées sur la production à un soutien direct aux revenus des agriculteurs.",
    "DE": "Fassen Sie den folgenden Text über die Gemeinsame Agrarpolitik der EU in höchstens 3 Sätzen zusammen und erfassen Sie die Hauptpunkte: Die Gemeinsame Agrarpolitik (GAP) ist die Agrarpolitik der Europäischen Union. Sie implementiert ein System von Agrarsubventionen und anderen Programmen. Sie wurde 1962 eingeführt und hat seitdem mehrere Reformen durchlaufen. Die GAP macht etwa ein Drittel des Gesamtbudgets der EU aus. Ihre Hauptziele sind die Verbesserung der landwirtschaftlichen Produktivität, die Sicherstellung eines angemessenen Lebensstandards für Landwirte, die Stabilisierung der Märkte, die Sicherstellung der Versorgung und die Gewährleistung angemessener Preise für Verbraucher. Jüngste Reformen konzentrierten sich auf mehr Umweltverträglichkeit der Landwirtschaft und die Umstellung von leistungsbezogenen Subventionen auf direkte Einkommensbeihilfen für Landwirte.",
    "EL": "Συνοψίστε το ακόλουθο κείμενο για την Κοινή Γεωργική Πολιτική της ΕΕ σε το πολύ 3 προτάσεις, συμπεριλαμβάνοντας τα βασικά σημεία: Η Κοινή Γεωργική Πολιτική (ΚΓΠ) είναι η γεωργική πολιτική της Ευρωπαϊκής Ένωσης. Εφαρμόζει ένα σύστημα γεωργικών επιδοτήσεων και άλλων προγραμμάτων. Εισήχθη το 1962 και έκτοτε έχει υποστεί αρκετές μεταρρυθμίσεις. Η ΚΓΠ αντιπροσωπεύει περίπου το ένα τρίτο του συνολικού προϋπολογισμού της ΕΕ. Οι κύριοι στόχοι της είναι η βελτίωση της γεωργικής παραγωγικότητας, η διασφάλιση ενός δίκαιου επιπέδου διαβίωσης για τους αγρότες, η σταθεροποίηση των αγορών, η διασφάλιση της διαθεσιμότητας εφοδίων και η διασφάλιση λογικών τιμών για τους καταναλωτές. Οι πρόσφατες μεταρρυθμίσεις έχουν εστιαστεί στην περιβαλλοντική βιωσιμότητα της γεωργίας και τη μετατόπιση των πληρωμών από επιδοτήσεις βάσει παραγωγής σε άμεση στήριξη εισοδήματος των αγροτών.",
    "HU": "Foglalja össze a következő szöveget az EU Közös Agrárpolitikájáról legfeljebb 3 mondatban, kiemelve a fő pontokat: A Közös Agrárpolitika (KAP) az Európai Unió mezőgazdasági politikája. Mezőgazdasági támogatások és egyéb programok rendszerét valósítja meg. 1962-ben vezették be és azóta számos reformon ment keresztül. A KAP az EU teljes költségvetésének körülbelül egyharmadát teszi ki. Fő céljai a mezőgazdasági termelékenység javítása, a gazdák méltányos életszínvonalának biztosítása, a piacok stabilizálása, a készletek rendelkezésre állásának biztosítása és a fogyasztók számára megfizethető árak biztosítása. A legutóbbi reformok a mezőgazdaság környezeti fenntarthatóságának növelésére és a támogatások termelésalapúról közvetlen jövedéktámogatásra való átállítására összpontosítottak.",
    "GA": "Achoimhrigh an téacs seo a leanas faoi Pholasaí Comhchoiteann Talmhaíochta an AE i 3 abairt ar a mhéad, ag breacadh na bpríomhphointí: Is é an Polasaí Comhchoiteann Talmhaíochta (PCT) polasaí talmhaíochta an Aontais Eorpaigh. Cuireann sé córas de shócmhainní talmhaíochta agus cláir eile i bhfeidhm. Cuireadh tús leis i 1962 agus tá sé tar éis roinnt leasuithe a fhulaing ó shin. Is cúigean den bhuiséad iomlán AE an PCT. Is iad a phríomhchuspóirí táirgiúlacht thalmhaíochta a fheabhsú, caighdeán maireachtála cothrom a chinntiú do fheirmeoirí, margaí a chobhsú, soláthar a chinntiú, agus praghsanna réasúnta a chinntiú do thomhaltóirí. Bhí an bhearthas le déanaí dírithe ar thalmhaíocht a dhéanamh níos inbhuanaithe go comhshaoil agus íocaíochtaí a aistriú ó shócmhainní bunaithe ar tháirgeacht go dtí tacaíocht ioncaim dhíreach do fheirmeoirí.",
    "IT": "Riassumi il seguente testo sulla Politica agricola comune dell'UE in al massimo 3 frasi, cogliendo i punti chiave: La Politica agricola comune (PAC) è la politica agricola dell'Unione europea. Implementa un sistema di sussidi agricoli e altri programmi. È stata introdotta nel 1962 e ha subito diverse riforme da allora. La PAC rappresenta circa un terzo del bilancio totale dell'UE. I suoi obiettivi principali sono migliorare la produttività agricola, garantire un equo tenore di vita agli agricoltori, stabilizzare i mercati, assicurare la disponibilità delle forniture e garantire prezzi ragionevoli per i consumatori. Le recenti riforme si sono concentrate sulla maggiore sostenibilità ambientale dell'agricoltura e sullo spostamento dei pagamenti dai sussidi basati sulla produzione al sostegno diretto del reddito degli agricoltori.",
    "LV": "Kopsavilkiet šo tekstu par ES Kopējo lauksaimniecības politiku ne vairāk kā 3 teikumos, iekļaujot galvenos punktus: Kopējā lauksaimniecības politika (KLP) ir Eiropas Savienības lauksaimniecības politika. Tā īsteno lauksaimniecības subsīdiju un citu programmu sistēmu. Tā tika ieviesta 1962. gadā un kopš tā laika ir veiktas vairākas reformas. KLP veido aptuveni trešdaļu no ES kopējā budžeta. Tās galvenie mērķi ir uzlabot lauksaimniecības produktivitāti, nodrošināt godīgu dzīves līmeni lauksaimniekiem, stabilizēt tirgus, nodrošināt piegādes pieejamību un nodrošināt saprātīgas cenas patērētājiem. Nesenās reformas ir koncentrējušās uz lauksaimniecības padarīšanu videi ilgtspējīgāku un maksājumu pārorientēšanu no ražošanas balstītām subsīdijām uz tiešiem ienākumu atbalstiem lauksaimniekiem.",
    "LT": "Apibendrinkite šį tekstą apie ES Bendrąją žemės ūkio politiką ne daugiau kaip 3 sakiniuose, nurodydami pagrindinius punktus: Bendroji žemės ūkio politika (BŽŪP) yra Europos Sąjungos žemės ūkio politika. Ji įgyvendina žemės ūkio subsidijų ir kitų programų sistemą. Ji buvo įvesta 1962 m. ir nuo to laiko patyrė keletą reformų. BŽŪP sudaro apie trečdalį ES bendrojo biudžeto. Pagrindiniai jos tikslai yra gerinti žemės ūkio produktyvumą, užtikrinti sąžiningą gyvenimo lygį ūkininkams, stabilizuoti rinkas, užtikrinti tiekimų prieinamumą ir užtikrinti pagrįstas kainas vartotojams. Naujausios reformos orientavosi į žemės ūkio ekologišką darną ir mokėjimų perkėlimą iš gamybos pagrįstų subsidijų tiesioginėms ūkininkų pajamų paramoms.",
    "MT": "Isummarizza t-test segwenti dwar il-Politika Agrikola Komuni tal-UE f'massimu 3 sentenzi, billi tinkludi l-punti ewlenin: Il-Politika Agrikola Komuni (PAK) hija l-politika agrikola tal-Unjoni Ewropea. Timplimenta sistema ta' sussidji agrikoli u programmi oħra. Ħarġet fl-1962 u għaddiet minn diversi riformi minn dakinhar. Il-PAK tirrappreżenta madwar terz tal-baġit totali tal-UE. L-għanijiet prinċipali tagħha huma li ttaffa l-produttività agrikola, tiżgura livell ġust tal-għajxien għall-bdiewa, tistabbilixxi s-swieq, tiżgura d-disponibbiltà tal-provvisti, u tiżgura prezzijiet raġonevoli għall-konsumaturi. Ir-riformi reċenti ffukaw fuq li tagħmel l-agrikoltura iktar sostenibbli ambjentalment u tibdel il-pagamenti minn sussidji bbażati fuq il-produzzjoni għal appoġġ dirett tad-dħul għall-bdiewa.",
    "PL": "Podsumuj poniższy tekst o Wspólnej Polityce Rolnej UE w maksymalnie 3 zdaniach, uwzględniając kluczowe punkty: Wspólna Polityka Rolna (WPR) jest polityką rolną Unii Europejskiej. Realizuje system subsydiów rolnych i innych programów. Wprowadzona została w 1962 roku i od tego czasu przeszła kilka reform. WPR stanowi około jednej trzeciej całkowitego budżetu UE. Jej główne cele to poprawa produktywności rolnictwa, zapewnienie rolnikom uczciwego poziomu życia, stabilizacja rynków, zapewnienie dostępności zaopatrzenia i zapewnienie rozsądnych cen dla konsumentów. Ostatnie reformy skupiły się na bardziej zrównoważonym środowiskowo rolnictwie i przesunięciu płatności z subsydiów opartych na produkcji na bezpośrednie wsparcie dochodów rolników.",
    "PT": "Resuma o seguinte texto sobre a Política Agrícola Comum da UE em no máximo 3 frases, captando os pontos-chave: A Política Agrícola Comum (PAC) é a política agrícola da União Europeia. Implementa um sistema de subsídios agrícolas e outros programas. Foi introduzida em 1962 e sofreu várias reformas desde então. A PAC representa cerca de um terço do orçamento total da UE. Seus principais objetivos são melhorar a produtividade agrícola, garantir um nível de vida justo para os agricultores, estabilizar os mercados, assegurar a disponibilidade de suprimentos e garantir preços razoáveis para os consumidores. As reformas recentes focaram em tornar a agricultura mais ambientalmente sustentável e mudar os pagamentos de subsídios baseados na produção para apoio direto à renda dos agricultores.",
    "RO": "Rezumați următorul text despre Politica Agricolă Comună a UE în cel mult 3 propoziții, surprinzând punctele cheie: Politica Agricolă Comună (PAC) este politica agricolă a Uniunii Europene. Implementează un sistem de subvenții agricole și alte programe. A fost introdusă în 1962 și a suferit mai multe reforme de atunci. PAC reprezintă aproximativ o treime din bugetul total al UE. Principalele sale obiective sunt îmbunătățirea productivității agricole, asigurarea unui standard de viață echitabil pentru fermieri, stabilizarea piețelor, asigurarea disponibilității aprovizionării și asigurarea unor prețuri rezonabile pentru consumatori. Reformele recente s-au concentrat pe creșterea sustenabilității de mediu a agriculturii și schimbarea plăților de la subvenții bazate pe producție la sprijin direct pentru venitul fermierilor.",
    "SK": "Zhrňte nasledujúci text o Spoločnej poľnohospodárskej politike EÚ v maximálne 3 vetách s uvedením kľúčových bodov: Spoločná poľnohospodárska politika (SPP) je poľnohospodárska politika Európskej únie. Uplatňuje systém poľnohospodárskych dotácií a iných programov. Bola zavedená v roku 1962 a odvtedy prešla niekoľkými reformami. SPP predstavuje asi tretinu celkového rozpočtu EÚ. Jej hlavné ciele sú zlepšiť poľnohospodársku produktivitu, zabezpečiť spravodlivú životnú úroveň pre poľnohospodárov, stabilizovať trhy, zabezpečiť dostupnosť dodávok a zabezpečiť primerané ceny pre spotrebiteľov. Nedávne reformy sa zamerali na ekologickejšie poľnohospodárstvo a presun platieb z dotácií založených na produkcii na priamu podporu príjmov poľnohospodárov.",
    "SL": "Povzemite naslednji besedilo o Skupni kmetijski politiki EU v največ 3 povedih z navedbo ključnih točk: Skupna kmetijska politika (SKP) je kmetijska politika Evropske unije. Uveljavlja sistem kmetijskih subvencij in drugih programov. Uvedena je bila leta 1962 in je od takrat doživela več reform. SKP predstavlja približno tretjino celotnega proračuna EU. Njeni glavni cilji so izboljšati kmetijsko produktivnost, zagotoviti pravično življenjsko raven kmetom, stabilizirati trge, zagotoviti razpoložljivost oskrbe in zagotoviti razumne cene za potrošnike. Nedavne reforme so se osredotočile na okoljsko trajnostnejše kmetijstvo in premik plačil s subvencij, ki temeljijo na proizvodnji, na neposredno podporo dohodka kmetov.",
    "ES": "Resuma el siguiente texto sobre la Política Agrícola Común de la UE en un máximo de 3 oraciones, captando los puntos clave: La Política Agrícola Común (PAC) es la política agrícola de la Unión Europea. Implementa un sistema de subvenciones agrícolas y otros programas. Se introdujo en 1962 y ha sufrido varias reformas desde entonces. La PAC representa aproximadamente un tercio del presupuesto total de la UE. Sus principales objetivos son mejorar la productividad agrícola, garantizar un nivel de vida justo para los agricultores, estabilizar los mercados, asegurar la disponibilidad de suministros y garantizar precios razonables para los consumidores. Las reformas recientes se han centrado en hacer la agricultura más sostenible medioambientalmente y cambiar los pagos de subvenciones basadas en la producción a apoyo directo a los ingresos de los agricultores.",
    "SV": "Sammanfatta följande text om EU:s gemensamma jordbrukspolitik i högst 3 meningar och ta med nyckelpunkterna: Den gemensamma jordbrukspolitiken (GJP) är Europeiska unionens jordbrukspolitik. Den genomför ett system av jordbruksstöd och andra program. Den infördes 1962 och har genomgått flera reformer sedan dess. GJP står för cirka en tredjedel av EU:s totala budget. Dess huvudsakliga mål är att förbättra jordbrukets produktivitet, säkerställa en skälig levnadsstandard för jordbrukare, stabilisera marknaderna, säkerställa tillgången på förnödenheter och säkerställa skäliga priser för konsumenterna. Nyligen genomförda reformer har fokuserat på att göra jordbruket mer miljömässigt hållbart och flytta betalningar från produktionsbaserade bidrag till direkt inkomststöd för jordbrukare.",
}

# Compile all questions
def get_all_questions():
    """Return all 5 questions translated to all 24 EU languages."""
    questions = {}
    for lang_code in EU_LANGUAGES.keys():
        questions[lang_code] = {
            "Q1_FACTUAL_KNOWLEDGE": Q1_TRANSLATIONS[lang_code],
            "Q2_LOGICAL_REASONING": Q2_TRANSLATIONS[lang_code],
            "Q3_INSTRUCTION_FOLLOWING": Q3_TRANSLATIONS[lang_code],
            "Q4_CULTURAL_NUANCE": Q4_TRANSLATIONS[lang_code],
            "Q5_SUMMARIZATION_ACCURACY": Q5_TRANSLATIONS[lang_code],
        }
    return questions


def get_question_metadata():
    """Return metadata for each question."""
    return {
        "Q1": {
            "id": "Q1_FACTUAL_KNOWLEDGE",
            "category": "Factual Knowledge & Reasoning",
            "difficulty": "medium",
            "expected_elements": ["Lisbon", "population", "1986", "Iberian Peninsula"],
        },
        "Q2": {
            "id": "Q2_LOGICAL_REASONING", 
            "category": "Logical Reasoning & Problem Solving",
            "difficulty": "medium",
            "expected_answer": 15,
        },
        "Q3": {
            "id": "Q3_INSTRUCTION_FOLLOWING",
            "category": "Instruction Following & Format Adherence",
            "difficulty": "easy",
            "required_format": "JSON",
            "required_keys": ["original_text", "translated_text", "target_language"],
        },
        "Q4": {
            "id": "Q4_CULTURAL_NUANCE",
            "category": "Cultural & Contextual Understanding",
            "difficulty": "hard",
            "min_language_examples": 3,
        },
        "Q5": {
            "id": "Q5_SUMMARIZATION_ACCURACY",
            "category": "Text Summarization & Information Extraction",
            "difficulty": "medium",
            "max_sentences": 3,
            "key_elements": ["1962", "one-third", "productivity", "sustainability"],
        },
    }


if __name__ == "__main__":
    # Print summary
    total_languages = len(EU_LANGUAGES)
    questions_per_language = 5
    total_runs = total_languages * questions_per_language
    
    print(f"Total EU Languages: {total_languages}")
    print(f"Total Questions per Language: {questions_per_language}")
    print(f"Total Evaluation Runs per Model: {total_runs}")
    print(f"\nLanguages: {', '.join([f'{code} ({info['name']})' for code, info in EU_LANGUAGES.items()])}")
