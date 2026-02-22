"""
EU-FarmBook Context-Based Evaluation - 24 EU Languages
Questions designed to test RAG capabilities with search result context
Real agriculture questions that farmers/researchers ask chatbots

Structure: Each question includes context from top 5 search results
Fields used from search results: title, subtitle, description, keywords, ko_content_flat
"""

# Note: CONTEXT_DATA will be populated with actual search results
# For each question, include top 5 search results

CONTEXT_DATA = {
    "Q1": [
        # TODO: Add 5 JSON objects from search results for Q1
    ],
    "Q2": [
        # TODO: Add 5 JSON objects from search results for Q2
    ],
    "Q3": [
        # TODO: Add 5 JSON objects from search results for Q3
    ],
    "Q4": [
        # TODO: Add 5 JSON objects from search results for Q4
    ],
    "Q5": [
        # TODO: Add 5 JSON objects from search results for Q5
    ],
}

# Q1: Organic Weed Control - Practical farmer question
Q1_TRANSLATIONS = {
    "BG": "Какви органични методи за борба с плевелите препоръчвате за зърнени култури в умерен климат? Искам алтернативи на хербицидите.",
    "HR": "Koje organske metode suzbijanja korova preporučujete za žitarice u umjerenoj klimi? Želim alternative herbicidima.",
    "CS": "Jaké organické metody hubení plevelů doporučujete pro obilniny v mírném podnebí? Chci alternativy k herbicidům.",
    "DA": "Hvilke organiske ukrudtsbekæmpelsesmetoder anbefaler du til kornafgrøder i tempereret klima? Jeg vil have alternativer til herbicider.",
    "NL": "Welke biologische onkruidbestrijdingsmethoden adviseert u voor granen in een gematigd klimaat? Ik wil alternatieven voor herbiciden.",
    "EN": "What organic weed control methods do you recommend for cereal crops in temperate climate? I want alternatives to herbicides.",
    "ET": "Milliseid orgaanilisi umbrohutõrje meetodeid soovitate teraviljadele parasvöötmes? Tahan alternatiive herbitsiididele.",
    "FI": "Mitä luomu rikkakasvien torjuntamenetelmiä suosittelet viljakasveille lauhkeassa ilmastossa? Haluan vaihtoehtoja herbisideille.",
    "FR": "Quelles méthodes de lutte biologique contre les adventices recommandez-vous pour les céréales en climat tempéré? Je veux des alternatives aux herbicides.",
    "DE": "Welche organischen Unkrautbekämpfungsmethoden empfehlen Sie für Getreide im gemäßigten Klima? Ich möchte Alternativen zu Herbiziden.",
    "EL": "Ποιες οργανικές μέθοδοι καταπολέμησης ζιζανίων συνιστώνται για δημητριακά σε εύκρατο κλίμα; Θέλω εναλλακτικές λύσεις στα ζιζανιοκτόνα.",
    "HU": "Milyen organikus gyomirtási módszereket ajánl gabonafélék számára mérsékelt éghajlaton? Alternatívákat szeretnék a gyomirtószerek helyett.",
    "GA": "Cad iad na modhanna orgánacha rialaithe luibheanna a mholann tú do chruithneacht i gclíomadh measartha? Teastaíonn roghanna eile uaim in ionad luibhicídí.",
    "IT": "Quali metodi biologici di controllo delle infestanti consigliate per i cereali in clima temperato? Voglio alternative agli erbicidi.",
    "LV": "Kādas organiskās nezāļu kontroles metodes jūs iesakat graudaugiem mērenā klimatā? Es gribu alternatīvas herbicīdiem.",
    "LT": "Kokias organines piktžolių kontrolės metodas rekomenduojate grūdiniams augalams vidutinio klimato sąlygomis? Noriu alternatyvų herbicidams.",
    "MT": "Liema metodi organiċi ta' kontroll tal-ħaxix ħazin tirrakkomanda għall-ġawhar f'klima temperata? Irid alternattivi għall-erbiċidi.",
    "PL": "Jakie organiczne metody zwalczania chwastów polecasz dla zbóż w klimacie umiarkowanym? Chcę alternatyw dla herbicydów.",
    "PT": "Quais métodos orgânicos de controle de ervas daninhas você recomenda para cereais em clima temperado? Quero alternativas aos herbicidas.",
    "RO": "Ce metode organice de combatere a buruienilor recomandați pentru cereale în climat temperat? Vreau alternative la erbicide.",
    "SK": "Aké organické metódy kontroly burín odporúčate pre obilniny v miernom podnebí? Chcem alternatívy k herbicídom.",
    "SL": "Katere organske metode nadzora plevelov priporočate za žitarice v zmernem podnebju? Želim alternative herbicidom.",
    "ES": "¿Qué métodos orgánicos de control de malezas recomiendas para cereales en clima templado? Quiero alternativas a los herbicidas.",
    "SV": "Vilka organiska ogräsbekämpningsmetoder rekommenderar du för spannmål i tempererat klimat? Jag vill ha alternativ till herbicider.",
}

# Q2: Soil Health Improvement - Soil management question
Q2_TRANSLATIONS = {
    "BG": "Как мога да подобря здравето на почвата и биоразнообразието на моята ферма? Кои практики имат доказан ефект?",
    "HR": "Kako mogu poboljšati zdravlje tla i biodiverzitet na svojoj farmi? Koje prakse imaju dokazani učinak?",
    "CS": "Jak mohu zlepšit zdraví půdy a biodiverzitu na své farmě? Které postupy mají prokázaný účinek?",
    "DA": "Hvordan kan jeg forbedre jordens sundhed og biodiversitet på min gård? Hvilke praksisser har dokumenteret effekt?",
    "NL": "Hoe kan ik de gezondheid van de bodem en de biodiversiteit op mijn boerderij verbeteren? Welke praktijken hebben een bewezen effect?",
    "EN": "How can I improve soil health and biodiversity on my farm? Which practices have proven effectiveness?",
    "ET": "Kuidas saan parandada oma talu mulla tervist ja elurikkust? Millistel tavadel on tõestatud tõhusus?",
    "FI": "Miten voin parantaa maaperän terveyttä ja biodiversiteettiä tilallani? Millä käytännöillä on todettu tehokkuus?",
    "FR": "Comment puis-je améliorer la santé des sols et la biodiversité dans mon exploitation? Quelles pratiques ont une efficacité prouvée?",
    "DE": "Wie kann ich die Bodengesundheit und Biodiversität auf meinem Hof verbessern? Welche Praktiken haben nachgewiesene Wirksamkeit?",
    "EL": "Πώς μπορώ να βελτιώσω την υγεία του εδάφους και τη βιοποικιλότητα στην εκμετάλλευσή μου; Ποιες πρακτικές έχουν αποδεδειγμένη αποτελεσματικότητα;",
    "HU": "Hogyan javíthatom a talaj egészségét és a biodiverzitást a gazdaságomban? Mely gyakorlatoknak van bizonyított hatékonyságuk?",
    "GA": "Conas is féidir liom sláinte ithir agus bithéagsúlacht a fheabhsú ar mo fheirm? Cad iad na cleachtais a bhfuil éifeachtúlacht chruthaithe acu?",
    "IT": "Come posso migliorare la salute del suolo e la biodiversità nella mia azienda? Quali pratiche hanno un'efficacia comprovata?",
    "LV": "Kā es varu uzlabot augsnes veselību un bioloģisko daudzveidību savā saimniecībā? Kurām praksēm ir pierādīta efektivitāte?",
    "LT": "Kaip galiu pagerinti dirvožemio sveikatą ir biologinę įvairovę savo ūkyje? Kurios praktikos turi įrodytą veiksmingumą?",
    "MT": "Kif nista' ntejjeb is-saħħa tal-ħamrija u d-diversità bijoloġika fir-ranch tiegħi? Liema prattiċi għandhom effettività ppruvata?",
    "PL": "Jak mogę poprawić zdrowie gleby i różnorodność biologiczną w moim gospodarstwie? Które praktyki mają udokumentowaną skuteczność?",
    "PT": "Como posso melhorar a saúde do solo e a biodiversidade na minha fazenda? Quais práticas têm eficácia comprovada?",
    "RO": "Cum pot îmbunătăți sănătatea solului și biodiversitatea în ferma mea? Ce practici au eficacitate dovedită?",
    "SK": "Ako môžem zlepšiť zdravie pôdy a biodiverzitu na svojej farme? Ktoré postupy majú preukázanú účinnosť?",
    "SL": "Kako lahko izboljšam zdravje tal in biotsko raznovrstnost na svoji kmetiji? Katere prakse imajo dokazano učinkovitost?",
    "ES": "¿Cómo puedo mejorar la salud del suelo y la biodiversidad en mi granja? ¿Qué prácticas tienen eficacia demostrada?",
    "SV": "Hur kan jag förbättra markhälsa och biologisk mångfald på min gård? Vilka metoder har bevisad effektivitet?",
}

# Q3: Climate Adaptation - Climate change question
Q3_TRANSLATIONS = {
    "BG": "Как да адаптирам земеделските си практики към променящия се климат в моя регион? Какви са последните проучвания за устойчиви методи?",
    "HR": "Kako prilagoditi svoje poljoprivredne prakse promjenjivoj klimi u mom području? Što kažu najnovija istraživanja o održivim metodama?",
    "CS": "Jak přizpůsobit své zemědělské postupy měnícímu se klimatu v mé oblasti? Co říkají nejnovější výzkumy o udržitelných metodách?",
    "DA": "Hvordan tilpasser jeg mine landbrugspraksisser til det skiftende klima i min region? Hvad siger den nyeste forskning om bæredygtige metoder?",
    "NL": "Hoe pas ik mijn landbouwpraktijken aan op het veranderende klimaat in mijn regio? Wat zegt het nieuwste onderzoek over duurzame methoden?",
    "EN": "How do I adapt my farming practices to the changing climate in my region? What does the latest research say about sustainable methods?",
    "ET": "Kuidas kohandada oma põllumajandustavasid muutuva kliimaga minu piirkonnas? Mida ütleb uusim uuring jätkusuutlike meetodite kohta?",
    "FI": "Miten sopeutan maatalouskäytäntöni alueeni muuttuvaan ilmastoon? Mitä uusin tutkimus sanoo kestävistä menetelmistä?",
    "FR": "Comment adapter mes pratiques agricoles au climat changeant dans ma région? Que dit la dernière recherche sur les méthodes durables?",
    "DE": "Wie passe ich meine landwirtschaftlichen Praktiken an den sich ändernden Klima in meiner Region an? Was sagt die neueste Forschung über nachhaltige Methoden?",
    "EL": "Πώς προσαρμόζω τις γεωργικές μου πρακτικές στο μεταβαλλόμενο κλίμα στην περιοχή μου; Τι λέει η πιο πρόσφατη έρευνα για τις βιώσιμες μεθόδους;",
    "HU": "Hogyan alakítsam át mezőgazdasági gyakorlataimat a változó éghajlathoz a régiómban? Mit mond a legfrissebb kutatás a fenntartható módszerekről?",
    "GA": "Conas is féidir liom mo chleachtais talmhaíochta a oiriúnú don aeráid ag athrú i mo réigiún? Cad a deir an taighde is déanaí faoi mhodhanna inbhuanaithe?",
    "IT": "Come posso adattare le mie pratiche agricole al clima che cambia nella mia regione? Cosa dice la ricerca più recente sui metodi sostenibili?",
    "LV": "Kā pielāgot savas lauksaimniecības prakses mainīgajam klimatam manā reģionā? Ko saka jaunākie pētījumi par ilgtspējīgām metodēm?",
    "LT": "Kaip pritaikyti savo žemės ūkio praktikas besikeičiančiam klimatui mano regione? Ką sako naujausi tyrimai apie tvarius metodus?",
    "MT": "Kif nista' naġġel il-prattiċi agrikoli tiegħi għall-klima li qed tinbidel fir-reġjun tiegħi? X'jgħid ir-riċerka l-aktar reċenti dwar il-metodi sostenibbli?",
    "PL": "Jak dostosować moje praktyki rolnicze do zmieniającego się klimatu w moim regionie? Co mówi najnowsze badanie o metodach zrównoważonych?",
    "PT": "Como adapto minhas práticas agrícolas às mudanças climáticas na minha região? O que diz a pesquisa mais recente sobre métodos sustentáveis?",
    "RO": "Cum îmi adaptez practicile agricole la clima în schimbare din regiunea mea? Ce spune cea mai recentă cercetare despre metodele durabile?",
    "SK": "Ako prispôsobiť svoje poľnohospodárske praktiky meniacemu sa podnebiu v mojom regióne? Čo hovorí najnovší výskum o udržateľných metódach?",
    "SL": "Kako prilagoditi svoje kmetijske prakse spreminjajočemu podnebju v moji regiji? Kaj pravi najnovejša raziskava o trajnostnih metodah?",
    "ES": "¿Cómo adapto mis prácticas agrícolas al clima cambiante en mi región? ¿Qué dice la investigación más reciente sobre métodos sostenibles?",
    "SV": "Hur anpassar jag mitt jordbruksarbete till det föränderliga klimatet i min region? Vad säger den senaste forskningen om hållbara metoder?",
}

# Q4: EU Funding/Grants - Policy/practical question
Q4_TRANSLATIONS = {
    "BG": "Какви финансови стимули и програми на ЕС са достъпни за фермери, които искат да преминат към агроекологични практики? Как мога да кандидатствам?",
    "HR": "Kakve financijske poticaje i programe EU-a imaju na raspolaganju poljoprivrednici koji žele prijeći na agroekološke prakse? Kako se mogu prijaviti?",
    "CS": "Jaké finanční pobídky a programy EU jsou k dispozici pro farmáře, kteří chtějí přejít na agroekologické postupy? Jak se mohu přihlásit?",
    "DA": "Hvilke finansielle incitamenter og EU-programmer er tilgængelige for landmænd, der ønsker at skifte til agroøkologiske praksisser? Hvordan ansøger jeg?",
    "NL": "Welke financiële stimuleringsmaatregelen en EU-programma's zijn beschikbaar voor boeren die willen overschakelen op agro-ecologische praktijken? Hoe kan ik me aanmelden?",
    "EN": "What financial incentives and EU programs are available for farmers who want to transition to agroecological practices? How can I apply?",
    "ET": "Milliseid finantsstimuleid ja EL programme on saadaval talupidajatele, kes soovivad üle minna agroökoloogilistele tavadele? Kuida saan kandideerida?",
    "FI": "Mitä taloudellisia kannustimia ja EU-ohjelmia on saatavilla viljelijöille, jotka haluavat siirtyä agroekologisiin käytäntöihin? Miten voin hakea?",
    "FR": "Quelles incitations financières et programmes de l'UE sont disponibles pour les agriculteurs qui souhaitent passer à des pratiques agroécologiques? Comment puis-je postuler?",
    "DE": "Welche finanziellen Anreize und EU-Programme stehen Landwirten zur Verfügung, die zu agroökologischen Praktiken wechseln möchten? Wie kann ich mich bewerben?",
    "EL": "Ποιες οικονομικές ενισχύσεις και προγράμματα της ΕΕ είναι διαθέσιμα για αγρότες που θέλουν να μεταβούν σε αγροοικολογικές πρακτικές; Πώς μπορώ να υποβάλω αίτηση;",
    "HU": "Milyen pénzügyi ösztönzők és EU-programok állnak rendelkezésre azoknak a gazdáknak, akik agroökológiai gyakorlatra akarnak áttérni? Hogyan pályázhatok?",
    "GA": "Cé na spreagthaí airgeadais agus cláir AE atá ar fáil d'fheirmeoirí ar mhaith leo aistriú chuig cleachtais agraieiceolaíocha? Conas is féidir liom iarratas a dhéanamh?",
    "IT": "Quali incentivi finanziari e programmi UE sono disponibili per gli agricoltori che desiderano passare a pratiche agroecologiche? Come posso candidarmi?",
    "LV": "Kādi finansiālie stimuli un ES programmas ir pieejamas zemniekiem, kuri vēlas pāriet uz agroekoloģiskām praksēm? Kā varu pieteikties?",
    "LT": "Kokios finansinės paskatos ir ES programos yra prieinamos ūkininkams, norintiems pereiti prie agroekologinių praktikų? Kaip galiu kreiptis?",
    "MT": "X'inċentivi finanjarji u programmi tal-UE huma disponibbli għall-bdiewa li jixtiequ jgħaddu għal prattiċi agroekoloġiċi? Kif nista' napplika?",
    "PL": "Jakie zachęty finansowe i programy UE są dostępne dla rolników, którzy chcą przejść na praktyki agroekologiczne? Jak mogę się ubiegać?",
    "PT": "Quais incentivos financeiros e programas da UE estão disponíveis para agricultores que desejam fazer a transição para práticas agroecológicas? Como posso me candidatar?",
    "RO": "Ce stimulente financiare și programe UE sunt disponibile pentru fermierii care doresc să treacă la practici agroecologice? Cum pot aplica?",
    "SK": "Aké finančné stimuly a programy EÚ sú dostupné pre farmárov, ktorí sa chcú presunúť na agroekologické praktiky? Ako sa môžem prihlásiť?",
    "SL": "Kakšne finančne spodbude in programi EU so na voljo kmetom, ki želijo preiti na agroekološke prakse? Kako se lahko prijavim?",
    "ES": "¿Qué incentivos financieros y programas de la UE están disponibles para agricultores que desean hacer la transición a prácticas agroecológicas? ¿Cómo puedo aplicar?",
    "SV": "Vilka ekonomiska incitament och EU-program finns tillgängliga för bönder som vill övergå till agroekologiska metoder? Hur kan jag ansöka?",
}

# Q5: Integrated Pest Management - Technical question
Q5_TRANSLATIONS = {
    "BG": "Какво представлява интегрираното управление на вредителите (IPM) и как мога да го приложа за контрол на неприятели по царевицата? Кои са основните принципи?",
    "HR": "Što je integrirani management štetočina (IPM) i kako ga mogu primijeniti za kontrolu štetočina na kukuruzu? Koji su osnovni principi?",
    "CS": "Co je integrovaná ochrana proti škůdcům (IPM) a jak ji mohu aplikovat na kontrolu škůdců kukuřice? Jaké jsou základní principy?",
    "DA": "Hvad er integreret skadedyrsbekæmpelse (IPM), og hvordan kan jeg anvende det til bekæmpelse af skadedyr i majs? Hvad er de grundlæggende principper?",
    "NL": "Wat is geïntegreerde plaagbeheersing (IPM) en hoe kan ik dit toepassen voor plaagbeheersing in maïs? Wat zijn de basisprincipes?",
    "EN": "What is Integrated Pest Management (IPM) and how can I apply it for pest control in maize? What are the basic principles?",
    "ET": "Mis on integreeritud kahjuritõrje (IPM) ja kuidas saan seda maisi kahjurite tõrjeks rakendada? Mis on põhiprintsiibid?",
    "FI": "Mikä on integroitu tuholaistorjunta (IPM) ja miten voin soveltaa sitä maissin tuholaistorjuntaan? Mitkä ovat perusperiaatteet?",
    "FR": "Qu'est-ce que la lutte intégrée contre les ravageurs (IPM) et comment puis-je l'appliquer pour la protection du maïs? Quels sont les principes de base?",
    "DE": "Was ist Integrierter Pflanzenschutz (IPM) und wie kann ich ihn zur Schädlingsbekämpfung bei Mais anwenden? Was sind die Grundprinzipien?",
    "EL": "Τι είναι η Ολοκληρωμένη Διαχείριση Εχθρών (IPM) και πώς μπορώ να την εφαρμόσω για τον έλεγχο εχθρών στο καλαμπόκι; Ποιές είναι οι βασικές αρχές;",
    "HU": "Mi az integrált kártevőgazdálkodás (IPM), és hogyan alkalmazhatom kukorica kártevők elleni védekezésre? Mik az alapelvek?",
    "GA": "Cad é Bainistíocht Comhtháite Aicídí (IPM) agus conas is féidir liom é a chur i bhfeidhm le haghaidh rialú aicídí ar arbhar? Cad iad na prionsabail bhunúsacha?",
    "IT": "Cos'è la Gestione Integrata dei Parassiti (IPM) e come posso applicarla per il controllo dei parassiti nel mais? Quali sono i principi di base?",
    "LV": "Kas ir integrētā kaitēkļu pārvaldība (IPM) un kā to var pielietot kukurūzas kaitēkļu kontrolei? Kādi ir pamatprincipi?",
    "LT": "Kas yra integruota kenkėjų valdymo sistema (IPM) ir kaip ją galima taikyti kukurūzų kenkėjų kontrolei? Kokie yra pagrindiniai principai?",
    "MT": "X'inhi l-Management Integrat tal-Annimali ħżiena (IPM) u kif nista' napplikaha għall-kontroll tal-annimali ħżiena fil-qamħ? X'inhu l-prinċipji bażiċi?",
    "PL": "Czym jest zintegrowana ochrona roślin (IPM) i jak mogę ją zastosować do ochrony kukurydzy przed szkodnikami? Jakie są podstawowe zasady?",
    "PT": "O que é o Manejo Integrado de Pragas (IPM) e como posso aplicá-lo para controle de pragas no milho? Quais são os princípios básicos?",
    "RO": "Ce este Managementul Integrat al Dăunătorilor (IPM) și cum îl pot aplica pentru controlul dăunătorilor la porumb? Care sunt principiile de bază?",
    "SK": "Čo je integrovaná ochrana proti škodcom (IPM) a ako ju môžem aplikovať na kontrolu škodcov kukurice? Aké sú základné princípy?",
    "SL": "Kaj je integrirani upravljanje škodljivcev (IPM) in kako ga lahko uporabim za nadzor škodljivcev v koruzi? Kakšna so osnovna načela?",
    "ES": "¿Qué es el Manejo Integrado de Plagas (IPM) y cómo puedo aplicarlo para el control de plagas en maíz? ¿Cuáles son los principios básicos?",
    "SV": "Vad är Integrerat skadedjursförsvar (IPM) och hur kan jag tillämpa det för skadedjursbekämpning i majs? Vilka är de grundläggande principerna?",
}

# Function to get complete question data with context
def get_context_questions():
    """
    Returns all 5 questions with their contexts.
    NOTE: CONTEXT_DATA needs to be populated with actual search results.
    """
    return {
        "Q1": {
            "question_translations": Q1_TRANSLATIONS,
            "context": CONTEXT_DATA.get("Q1", []),
        },
        "Q2": {
            "question_translations": Q2_TRANSLATIONS,
            "context": CONTEXT_DATA.get("Q2", []),
        },
        "Q3": {
            "question_translations": Q3_TRANSLATIONS,
            "context": CONTEXT_DATA.get("Q3", []),
        },
        "Q4": {
            "question_translations": Q4_TRANSLATIONS,
            "context": CONTEXT_DATA.get("Q4", []),
        },
        "Q5": {
            "question_translations": Q5_TRANSLATIONS,
            "context": CONTEXT_DATA.get("Q5", []),
        },
    }


# Example format for CONTEXT_DATA after population:
# CONTEXT_DATA = {
#     "Q1": [
#         {
#             "title": "...",
#             "subtitle": "...",
#             "description": "...",
#             "keywords": ["..."],
#             "ko_content_flat": "..."
#         },
#         # ... 4 more objects
#     ],
#     ...
# }
