"""
EU-FarmBook Context-Based Evaluation - 24 EU Languages
Questions designed to test RAG capabilities with search result context

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

# Q1: Context Synthesis - Summarize main findings across sources
Q1_TRANSLATIONS = {
    "BG": "На база на предоставените документи, обобщете основните констатации за агроекологичните практики за контрол на плевелите в Полша. Кои методи са най-често използвани и с каква ефективност?",
    "HR": "Na temelju pruženih dokumenata, sažmite ključne nalaze o agroekološkim praksama za kontrolu korova u Poljskoj. Koje se metode najčešće koriste i koliko su učinkovite?",
    "CS": "Na základě poskytnutých dokumentů shrňte klíčové zjištění o agroekologických praxích pro kontrolu plevelů v Polsku. Které metody se nejčastěji používají a jak jsou efektivní?",
    "DA": "Baseret på de leverede dokumenter, opsummer de vigtigste resultater om agroøkologiske praksisser til ukrudtskontrol i Polen. Hvilke metoder bruges oftest, og hvor effektive er de?",
    "NL": "Op basis van de verstrekte documenten, vat de belangrijkste bevindingen samen over agro-ecologische praktijken voor onkruidbestrijding in Polen. Welke methoden worden het meest gebruikt en hoe effectief zijn ze?",
    "EN": "Based on the provided documents, summarize the key findings about agroecological practices for weed control in Poland. Which methods are most commonly used and how effective are they?",
    "ET": "Esitatud dokumentide põhjal võtke kokku peamised järeldused agroökoloogiliste tavade kohta umbrohutõrjes Poolas. Milliseid meetodeid kasutatakse kõige sagedamini ja kui tõhusad need on?",
    "FI": "Toimitettujen asiakirjojen perusteella tiivistä tärkeimmät havainnot agroekologisista käytännöistä rikkakasvien torjunnassa Puolassa. Mitä menetelmiä käytetään yleisimmin ja kuinka tehokkaita ne ovat?",
    "FR": "Sur la base des documents fournis, résumez les principales conclusions concernant les pratiques agroécologiques pour le contrôle des adventices en Pologne. Quelles méthodes sont les plus utilisées et à quel point sont-elles efficaces?",
    "DE": "Basierend auf den bereitgestellten Dokumenten fassen Sie die wichtigsten Erkenntnisse über agroökologische Praktiken zur Unkrautbekämpfung in Polen zusammen. Welche Methoden werden am häufigsten verwendet und wie effektiv sind sie?",
    "EL": "Με βάση τα παρεχόμενα έγγραφα, συνοψίστε τα βασικά ευρήματα σχετικά με τις αγροοικολογικές πρακτικές για τον έλεγχο των ζιζανίων στην Πολωνία. Ποιες μέθοδοι χρησιμοποιούνται πιο συχνά και πόσο αποτελεσματικές είναι?",
    "HU": "A rendelkezésre álló dokumentumok alapján foglalja össze a lengyelországi gyomirtás agroökológiai gyakorlatára vonatkozó legfontosabb megállapításokat. Mely módszereket használják leggyakrabban, és milyen hatékonyak?",
    "GA": "Bunaithe ar na doiciméid a cuireadh ar fáil, achoimreigh na príomhthorthaí maidir le cleachtais agraieiceolaíocha chun rialú luibheanna a dhéanamh sa Pholainn. Cé na modhanna a úsáidtear is minice agus cé chomh héifeachtach is atá siad?",
    "IT": "Sulla base dei documenti forniti, riassumi i risultati chiave sulle pratiche agroecologiche per il controllo delle infestanti in Polonia. Quali metodi sono più comunemente utilizzati e quanto sono efficaci?",
    "LV": "Pamantojoties uz sniegtajiem dokumentiem, apkopojiet galvenos secinājumus par agroekoloģiskajām praksēm nezāļu kontrolei Polijā. Kuras metodes visbiežāk izmanto un cik tās ir efektīvas?",
    "LT": "Remiantis pateiktais dokumentais, apibendrinkite pagrindines išvadas apie agroekologines praktikas piktžolių kontrolei Lenkijoje. Kurie metodai dažniausiai naudojami ir kokie jie efektyvūs?",
    "MT": "Bbażat fuq id-dokumenti pprovduti, issummarja l-konklużjonijiet ewlenin dwar il-prattiki agroekoloġiċi għall-kontroll tal-ħaxix ħazin fil-Polonja. Liema metodi jintużaw l-iktar u kif huma effettivi?",
    "PL": "Na podstawie dostarczonych dokumentów podsumuj kluczowe wnioski dotyczące praktyk agroekologicznych w zwalczaniu chwastów w Polsce. Jakie metody są najczęściej stosowane i jak skuteczne są?",
    "PT": "Com base nos documentos fornecidos, resuma as principais conclusões sobre práticas agroecológicas para controle de ervas daninhas na Polônia. Quais métodos são mais comumente usados e quão eficazes são?",
    "RO": "Pe baza documentelor furnizate, rezumați constatările cheie despre practicile agroecologice pentru controlul buruienilor în Polonia. Ce metode sunt cel mai frecvent utilizate și cât de eficiente sunt?",
    "SK": "Na základe poskytnutých dokumentov zhrňte kľúčové zistenia o agroekologických praxiach na kontrolu burín v Poľsku. Ktoré metódy sa najčastejšie používajú a aké sú efektívne?",
    "SL": "Na podlagi predloženih dokumentov povzemite ključne ugotovitve o agroekoloških praksah za nadzor plevelov na Poljskem. Katere metode se najpogosteje uporabljajo in kako učinkovite so?",
    "ES": "Basándose en los documentos proporcionados, resuma los hallazgos clave sobre prácticas agroecológicas para el control de malezas en Polonia. ¿Qué métodos se utilizan más comúnmente y qué tan efectivos son?",
    "SV": "Baserat på de tillhandahållna dokumenten, sammanfatta de viktigaste resultaten om agroekologiska metoder för bekämpning av ogräs i Polen. Vilka metoder används oftast och hur effektiva är de?",
}

# Q2: Information Extraction - Extract specific data points
Q2_TRANSLATIONS = {
    "BG": "Извлечете конкретни данни от документите: Какъв процент от фермерите използват сеитбообращение? Колко процента използват сертифициран посевен материал? Кои са споменатите алтернативни методи и с каква честота се използват?",
    "HR": "Izdvojite specifične podatke iz dokumenata: Koliki postotak poljoprivrednika koristi rotaciju usjeva? Koliko posto koristi certificirani sjemenski materijal? Koje su spomenute alternativne metode i koliko se često koriste?",
    "CS": "Z dokumentů extrahujte konkrétní údaje: Jaké procento farmářů používá střídání plodin? Kolik procent používá certifikovaný osivový materiál? Jaké jsou zmíněné alternativní metody a jak často se používají?",
    "DA": "Uddrag specifikke data fra dokumenterne: Hvilken procentdel af landmændene bruger afgrøderotation? Hvor mange procent bruger certificeret frømateriale? Hvilke alternative metoder nævnes, og hvor ofte bruges de?",
    "NL": "Haal specifieke gegevens uit de documenten: Welk percentage boeren gebruikt vruchtwisseling? Hoeveel procent gebruikt gecertificeerd pootgoed? Welke alternatieve methoden worden genoemd en hoe vaak worden ze gebruikt?",
    "EN": "Extract specific data from the documents: What percentage of farmers use crop rotation? What percentage use certified seed material? What are the mentioned alternative methods and how frequently are they used?",
    "ET": "Ekstraktige dokumentidest konkreetsed andmed: Milline protsent talupidajatest kasutab põllukultuuride vaheldust? Milline protsent kasutab sertifitseeritud seemnekorra? Millised on mainitud alternatiivsed meetodid ja kui sageli neid kasutatakse?",
    "FI": "Tee tiettyjä tietoja asiakirjoista: Kuinka suuri prosenttiosuus viljelijöistä käyttää viljelykiertoa? Kuinka monta prosenttia käyttää sertifioitua siemenmateriaalia? Mitä vaihtoehtoisia menetelmiä mainitaan ja kuinka usein niitä käytetään?",
    "FR": "Extrayez des données spécifiques des documents: Quel pourcentage d'agriculteurs utilisent la rotation des cultures? Quel pourcentage utilise du matériel de semences certifié? Quelles sont les méthodes alternatives mentionnées et à quelle fréquence sont-elles utilisées?",
    "DE": "Extrahieren Sie spezifische Daten aus den Dokumenten: Welcher Prozentsatz der Landwirte nutzt Fruchtwechsel? Wie viel Prozent verwenden zertifiziertes Saatgut? Welche alternativen Methoden werden erwähnt und wie häufig werden sie verwendet?",
    "EL": "Εξαγάγετε συγκεκριμένα δεδομένα από τα έγγραφα: Τι ποσοστό αγροτών χρησιμοποιεί εναλλαγή καλλιεργειών; Τι ποσοστό χρησιμοποιεί πιστοποιημένο σποροφόρο υλικό; Ποιές είναι οι αναφερόμενες εναλλακτικές μέθοδοι και πόσο συχνά χρησιμοποιούνται;",
    "HU": "Vonjon ki konkrét adatokat a dokumentumokból: A gazdák hány százaléka használ vetésforgót? Hány százalék használ tanúsított vetőmagot? Mik az említett alternatív módszerek és milyen gyakran használják őket?",
    "GA": "Bain sonraíochtaí sonracha amach as na doiciméid: Cé céatadán feirmeoirí a úsáideann rothlaíocht barraí? Cé mhéad a úsáideann ábhar síolaithe deimhnithe? Cad iad na modhanna malartacha a luadh agus cé chomh minic a úsáidtear iad?",
    "IT": "Estrai dati specifici dai documenti: Quale percentuale di agricoltori utilizza la rotazione delle colture? Quale percentuale utilizza materiale di semi certificato? Quali sono i metodi alternativi menzionati e con quale frequenza vengono utilizzati?",
    "LV": "Izdodiet konkrētus datus no dokumentiem: Kāds procents zemnieku izmanto augsekv maiņu? Cik procenti izmanto sertificēto sēklu materiālu? Kuras ir minētās alternatīvās metodes un cik bieži tās tiek izmantotas?",
    "LT": "Iš dokumentų išskirkite konkrečius duomenis: Koks procentas ūkininkų naudoja pasėlių sėjomainą? Kiek procentų naudoja sertifikuotą sėklinę medžiagą? Kokie yra minimi alternatyvūs metodai ir kaip dažnai jie naudojami?",
    "MT": "Estratti data speċifika mid-dokumenti: X'inhu l-perċentwal tal-bdiewa li jużaw ir-rotazzjoni tal-uċuh? X'inhu l-perċentwal li jużaw materjal ta' ġerq certifikat? X'in huma l-metodi alternattivi imsemmija u kif jintużaw?",
    "PL": "Wyodrębnij konkretne dane z dokumentów: Jaki procent rolników stosuje płodozmian? Ile procent używa certyfikowanego materiału siewnego? Jakie są wymienione metody alternatywne i jak często są stosowane?",
    "PT": "Extraia dados específicos dos documentos: Qual porcentagem de agricultores usa rotação de culturas? Qual porcentagem usa material de sementes certificado? Quais são os métodos alternativos mencionados e com que frequência são usados?",
    "RO": "Extrageți date specifice din documente: Ce procent de fermieri folosesc rotirea culturilor? Ce procent folosesc material săditor certificat? Care sunt metodele alternative menționate și cât de frecvent sunt utilizate?",
    "SK": "Z dokumentov extrahujte konkrétne údaje: Aké percento farmárov používa striedanie plodín? Koľko percent používa certifikovaný osivový materiál? Aké sú spomenuté alternatívne metódy a ako často sa používajú?",
    "SL": "Iz dokumentov izvlecite specifične podatke: Kakšen odstotek kmetov uporablja kolobarjenje? Koliko odstotkov uporablja certificirani semenski material? Katere so omenjene alternativne metode in kako pogosto se uporabljajo?",
    "ES": "Extraiga datos específicos de los documentos: ¿Qué porcentaje de agricultores usa rotación de cultivos? ¿Qué porcentaje usa material de semillas certificado? ¿Cuáles son los métodos alternativos mencionados y con qué frecuencia se usan?",
    "SV": "Extrahera specifika data från dokumenten: Vilken procentandel av bönderna använder växelbruk? Hur många procent använder certifierat frömaterial? Vilka är de nämnda alternativa metoderna och hur ofta används de?",
}

# Q3: Comparison - Compare across sources
Q3_TRANSLATIONS = {
    "BG": "Сравнете информацията от различните документи: Има ли разлики в препоръчваните практики? Кои методи са последователно подкрепени в множество източници, а кои са споменати само веднъж? Какво може да обясни тези разлики?",
    "HR": "Usporedite podatke iz različitih dokumenata: Postoje li razlike u preporučenim praksama? Koje metode su dosljedno podržane u više izvora, a koje su spomenute samo jednom? Što može objasniti te razlike?",
    "CS": "Porovnejte informace z různých dokumentů: Jsou nějaké rozdíly v doporučených postupech? Které metody jsou konzistentně podporovány ve více zdrojích a které jsou zmíněny pouze jednou? Co může vysvětlit tyto rozdíly?",
    "DA": "Sammenlign oplysningerne fra de forskellige dokumenter: Er der nogen forskelle i de anbefalede praksisser? Hvilke metoder understøttes konsekvent på tværs af flere kilder, og hvilke nævnes kun én gang? Hvad kan forklare disse forskelle?",
    "NL": "Vergelijk de informatie uit de verschillende documenten: Zijn er verschillen in de aanbevolen praktijken? Welke methoden worden consistent ondersteund in meerdere bronnen, en welke worden slechts één keer genoemd? Wat kan deze verschillen verklaren?",
    "EN": "Compare information from the different documents: Are there any differences in the recommended practices? Which methods are consistently supported across multiple sources, and which are mentioned only once? What might explain these differences?",
    "ET": "Võrrelge teavet erinevatest dokumentidest: Kas soovitatavates tavades on mingeid erinevusi? Milliseid meetodeid toetatakse järjekindlalt mitmes allikas ja milliseid mainitakse ainult üks kord? Mida võiksid need erinevused selgitada?",
    "FI": "Vertaa tietoja eri asiakirjoista: Onko suositelluissa käytännöissä eroja? Mitkä menetelmät ovat johdonmukaisesti tuettuja useissa lähteissä, ja mitkä mainitaan vain kerran? Mikä voisi selittää nämä erot?",
    "FR": "Comparez les informations des différents documents: Y a-t-il des différences dans les pratiques recommandées? Quelles méthodes sont systématiquement soutenues dans plusieurs sources, et lesquelles ne sont mentionnées qu'une seule fois? Que peut expliquer ces différences?",
    "DE": "Vergleichen Sie die Informationen aus den verschiedenen Dokumenten: Gibt es Unterschiede in den empfohlenen Praktiken? Welche Methoden werden über mehrere Quellen hinweg konsequent unterstützt, und welche werden nur einmal erwähnt? Was könnte diese Unterschiede erklären?",
    "EL": "Συγκρίνετε τις πληροφορίες από τα διαφορετικά έγγραφα: Υπάρχουν διαφορές στις προτεινόμενες πρακτικές; Ποιες μέθοδοι υποστηρίζονται συνεπώς σε πολλαπλές πηγές και ποιες αναφέρονται μόνο μία φορά; Τι θα μπορούσε να εξηγήσει αυτές τις διαφορές;",
    "HU": "Hasonlítsa össze az információkat a különböző dokumentumokból: Vannak-e különbségek az ajánlott gyakorlatokban? Mely módszereket támogatják következetesen több forrásban, és melyeket említenek csak egyszer? Mi magyarázhatja ezeket a különbségeket?",
    "GA": "Cuir an t-eolas ó na doiciméid éagsúla i gcomparáid: An bhfuil aon difríochtaí sna cleachtais a mholadh? Cé na modhanna a dtacaítear leo go leanúnach trasna foinsí éagsúla, agus cé na cinn a luadh uair amháin? Cad a d'fhéadfadh na difríochtaí seo a mhíniú?",
    "IT": "Confronta le informazioni dai diversi documenti: Ci sono differenze nelle pratiche raccomandate? Quali metodi sono costantemente supportati in più fonti, e quali sono menzionati solo una volta? Cosa potrebbe spiegare queste differenze?",
    "LV": "Salīdziniet informāciju no dažādiem dokumentiem: Vai ir kādas atšķirības ieteicamajās praksēs? Kuras metodes konsekventi atbalsta vairākos avotos, un kuras piemin tikai vienu reizi? Kas varētu izskaidrot šīs atšķirības?",
    "LT": "Palyginkite informaciją iš skirtingų dokumentų: Ar yra kokių nors skirtumų rekomenduojamoje praktikoje? Kurie metodai nuosekliai palaikomi keliuose šaltiniuose, o kurie minimi tik kartą? Ką galėtų paaiškinti šie skirtumai?",
    "MT": "Qabbel l-informazzjoni mid-dokumenti differenti: Hemm xi differenzi fil-prattiki rakkomandati? Liema metodi huma konsistentementi appoġġati f'sorsi multipli, u liema huma imsemmija darba biss? X'inhu li jista' jispjega dawn id-differenzi?",
    "PL": "Porównaj informacje z różnych dokumentów: Czy istnieją jakiekolwiek różnice w zalecanych praktykach? Które metody są konsekwentnie wspierane w wielu źródłach, a które są wymienione tylko raz? Co może wyjaśniać te różnice?",
    "PT": "Compare as informações dos diferentes documentos: Há alguma diferença nas práticas recomendadas? Quais métodos são consistentemente apoiados em várias fontes, e quais são mencionados apenas uma vez? O que pode explicar essas diferenças?",
    "RO": "Comparați informațiile din diferitele documente: Există diferențe în practicile recomandate? Ce metode sunt susținute constant în mai multe surse, și care sunt menționate doar o dată? Ce ar putea explica aceste diferențe?",
    "SK": "Porovnajte informácie z rôznych dokumentov: Sú nejaké rozdiely v odporúčaných postupoch? Ktoré metódy sú konsekventne podporované vo viacerých zdrojoch a ktoré sú spomenuté len raz? Čo by mohlo vysvetliť tieto rozdiely?",
    "SL": "Primerjajte informacije iz različnih dokumentov: Ali obstajajo kakršne koli razlike v priporočenih praksah? Katere metode so dosledno podprte v več virih in katere so omenjene samo enkrat? Kaj bi lahko razložilo te razlike?",
    "ES": "Compare la información de los diferentes documentos: ¿Hay alguna diferencia en las prácticas recomendadas? ¿Qué métodos son consistentemente apoyados en múltiples fuentes, y cuáles se mencionan solo una vez? ¿Qué podría explicar estas diferencias?",
    "SV": "Jämför informationen från de olika dokumenten: Finns det några skillnader i de rekommenderade metoderna? Vilka metoder stöds konsekvent över flera källor, och vilka nämns bara en gång? Vad skulle kunna förklara dessa skillnader?",
}

# Q4: Recommendation Synthesis - What would you recommend?
Q4_TRANSLATIONS = {
    "BG": "Въз основа на всички предоставени документи, какви конкретни препоръки бихте дали на полски фермер, който иска да подобри контрола на плевелите? Подкрепете препоръките си с доказателства от контекста.",
    "HR": "Na temelju svih pruženih dokumenata, koje specifične preporuke biste dali poljskom poljoprivredniku koji želi poboljšati kontrolu korova? Poduprite svoje preporuke dokazima iz konteksta.",
    "CS": "Na základě všech poskytnutých dokumentů, jaká konkrétní doporučení byste dali polskému farmáři, který chce zlepšit kontrolu plevelů? Podpořte svá doporučení důkazy z kontextu.",
    "DA": "Baseret på alle de leverede dokumenter, hvilke specifikke anbefalinger ville du give en polsk landmand, der ønsker at forbedre ukrudtskontrollen? Understøt dine anbefalinger med beviser fra konteksten.",
    "NL": "Op basis van alle verstrekte documenten, welke specifieke aanbevelingen zou u geven aan een Poolse boer die de onkruidbestrijding wil verbeteren? Ondersteun uw aanbevelingen met bewijs uit de context.",
    "EN": "Based on all the provided documents, what specific recommendations would you give to a Polish farmer who wants to improve weed control? Support your recommendations with evidence from the context.",
    "ET": "Kõigi esitatud dokumentide põhjal, milliseid konkreetseid soovitusi annaksite Poola talupidajale, kes soovib parandada umbrohutõrjet? Toetage oma soovitusi tõenditega kontekstist.",
    "FI": "Kaikkien toimitettujen asiakirjojen perusteella, mitä erityisiä suosituksia antaisit puolalaiselle viljelijälle, joka haluaa parantaa rikkakasvien torjuntaa? Tuke suosituksiasi todisteilla kontekstista.",
    "FR": "Sur la base de tous les documents fournis, quelles recommandations spécifiques donneriez-vous à un agriculteur polonais qui souhaite améliorer le contrôle des adventices? Appuyez vos recommandations sur des preuves du contexte.",
    "DE": "Basierend auf allen bereitgestellten Dokumenten, welche spezifischen Empfehlungen würden Sie einem polnischen Landwirt geben, der die Unkrautbekämpfung verbessern möchte? Untermauern Sie Ihre Empfehlungen mit Belegen aus dem Kontext.",
    "EL": "Με βάση όλα τα παρεχόμενα έγγραφα, ποιες συγκεκριμένες συστάσεις θα δίνατε σε έναν Πολωνό αγρότη που θέλει να βελτιώσει τον έλεγχο των ζιζανίων; Υποστηρίξτε τις συστάσεις σας με αποδεικτικά στοιχεία από το περιεχόμενο.",
    "HU": "Az összes rendelkezésre álló dokumentum alapján milyen konkrét ajánlásokat tenne egy lengyel gazdának, aki javítani szeretné a gyomirtást? Támogassa ajánlásait bizonyítékokkal a kontextusból.",
    "GA": "Bunaithe ar na doiciméid go léir a cuireadh ar fáil, cad iad na moltaí sonracha a thabharfá d'fheirmeoir Polannach ar mhaith leis an rialú luibheanna a fheabhsú? Tacaigh do mholtaí le fianaise ón gcomhthéacs.",
    "IT": "Sulla base di tutti i documenti forniti, quali raccomandazioni specifiche dareste a un agricoltore polacco che vuole migliorare il controllo delle infestanti? Sostieni le tue raccomandazioni con prove dal contesto.",
    "LV": "Pamantojoties uz visiem sniegtajiem dokumentiem, kādus konkrētus ieteikumus jūs dotu Polijas zemniekam, kurš vēlas uzlabot nezāļu kontroli? Atbalstiet savus ieteikumus ar pierādījumiem no konteksta.",
    "LT": "Remiantis visais pateiktais dokumentais, kokius konkrečius pasiūlymus pateiktumėte Lenkijos ūkininkui, norinčiam pagerinti piktžolių kontrolę? Pateikite savo rekomendacijų įrodymus iš konteksto.",
    "MT": "Bbażat fuq id-dokumenti kollha pprovduti, liema rakkomandazzjonijiet speċifiċi tagħti lill-bdiewa Pollak li jixtieq jitjib il-kontroll tal-ħaxix ħazin? Appoġġja r-rakkomandazzjonijiet tiegħek b'evidenza mill-kuntest.",
    "PL": "Na podstawie wszystkich dostarczonych dokumentów, jakie konkretne zalecenia dałbyś polskiemu rolnikowi, który chce poprawić zwalczanie chwastów? Uzasadnij swoje zalecenia dowodami z kontekstu.",
    "PT": "Com base em todos os documentos fornecidos, quais recomendações específicas você daria a um agricultor polonês que quer melhorar o controle de ervas daninhas? Apoie suas recomendações com evidências do contexto.",
    "RO": "Pe baza tuturor documentelor furnizate, ce recomandări specifice ați da unui fermier polonez care dorește să îmbunătățească controlul buruienilor? Susțineți-vă recomandările cu dovezi din context.",
    "SK": "Na základe všetkých poskytnutých dokumentov, aké konkrétne odporúčania by ste dali poľskému farmárovi, ktorý chce zlepšiť kontrolu burín? Podložte svoje odporúčania dôkazmi z kontextu.",
    "SL": "Na podlagi vseh predloženih dokumentov, kakšne konkretne priporočili bi dali poljskemu kmetu, ki želi izboljšati nadzor plevelov? Podprite svoje priporočilo z dokazi iz konteksta.",
    "ES": "Basándose en todos los documentos proporcionados, ¿qué recomendaciones específicas daría a un agricultor polaco que quiere mejorar el control de malezas? Apoye sus recomendaciones con evidencia del contexto.",
    "SV": "Baserat på alla tillhandahållna dokument, vilka specifika rekommendationer skulle du ge en polsk bonde som vill förbättra ogräsbekämpningen? Stöd dina rekommendationer med bevis från sammanhanget.",
}

# Q5: Critical Analysis - Identify gaps or limitations
Q5_TRANSLATIONS = {
    "BG": "Проанализирайте предоставените документи критично: Каква информация липсва, която би била полезна за вземане на решение? Какви ограничения има в представените данни? Какви допълнителни изследвания или данни биха били необходими за по-добро разбиране на темата?",
    "HR": "Kritički analizirajte pružene dokumente: Koje informacije nedostaju koje bi bile korisne za donošenje odluka? Kakva su ograničenja u predstavljenim podacima? Koja dodatna istraživanja ili podaci bi bili potrebni za bolje razumijevanje teme?",
    "CS": "Kriticky analyzujte poskytnuté dokumenty: Jaké informace chybí, které by byly užitečné pro rozhodování? Jaká jsou omezení v prezentovaných datech? Jaké další výzkumy nebo data by byly potřebné pro lepší porozumění tématu?",
    "DA": "Analyser de leverede dokumenter kritisk: Hvilke oplysninger mangler, der ville være nyttige for beslutningstagning? Hvilke begrænsninger er der i de præsenterede data? Hvilke yderligere undersøgelser eller data ville være nødvendige for bedre at forstå emnet?",
    "NL": "Analyseer de verstrekte documenten kritisch: Welke informatie ontbreekt die nuttig zou zijn voor besluitvorming? Wat zijn de beperkingen van de gepresenteerde gegevens? Welk aanvullend onderzoek of welke gegevens zouden nodig zijn om het onderwerp beter te begrijpen?",
    "EN": "Critically analyze the provided documents: What information is missing that would be useful for decision-making? What are the limitations in the presented data? What additional research or data would be needed for a better understanding of the topic?",
    "ET": "Analüüsige kriitiliselt esitatud dokumente: Millist teavet on puudu, mis oleks otsuste tegemisel kasulik? Mis on esitatud andmete piirangud? Millised täiendavad uuringud või andmed oleksid vajalikud teema paremaks mõistmiseks?",
    "FI": "Analysoi toimitettuja asiakirjoja kriittisesti: Mitä tietoja puuttuu, jotka olisivat hyödyllisiä päätöksenteossa? Mitä rajoituksia esitetyissä tiedoissa on? Mitä lisätutkimuksia tai tietoja tarvittaisiin aiheen paremman ymmärtämiseksi?",
    "FR": "Analysez de manière critique les documents fournis: Quelles informations manquent qui seraient utiles pour la prise de décision? Quelles sont les limites des données présentées? Quelles recherches ou données supplémentaires seraient nécessaires pour une meilleure compréhension du sujet?",
    "DE": "Analysieren Sie die bereitgestellten Dokumente kritisch: Welche Informationen fehlen, die für die Entscheidungsfindung nützlich wären? Was sind die Einschränkungen der präsentierten Daten? Welche zusätzlichen Forschungen oder Daten wären für ein besseres Verständnis des Themas erforderlich?",
    "EL": "Αναλύστε κριτικά τα παρεχόμενα έγγραφα: Ποιες πληροφορίες λείπουν που θα ήταν χρήσιμες για τη λήψη αποφάσεων; Ποιοι είναι οι περιορισμοί στα παρουσιαζόμενα δεδομένα; Τι επιπλέον έρευνα ή δεδομένα θα χρειάζονταν για καλύτερη κατανόηση του θέματος;",
    "HU": "Elemezze kritikusan a rendelkezésre álló dokumentumokat: Milyen információk hiányoznak, amelyek hasznosak lennének a döntéshozatalban? Mik a bemutatott adatok korlátai? Milyen további kutatásokra vagy adatokra lenne szükség a téma jobb megértéséhez?",
    "GA": "Déan anailís chriticiúil ar na doiciméid a cuireadh ar fáil: Cad é an t-eolas atá ar iarraidh a bheadh úsáideach le haghaidh cinnteoireachta? Cad iad na teorainneacha sna sonraí a cuireadh i láthair? Cad é an taighde breise nó na sonraí a bheadh ag teastáil le tuiscint níos fearr a fháil ar an ábhar?",
    "IT": "Analizza criticamente i documenti forniti: Quali informazioni mancano che sarebbero utili per il processo decisionale? Quali sono le limitazioni nei dati presentati? Quali ricerche o dati aggiuntivi sarebbero necessari per una migliore comprensione dell'argomento?",
    "LV": "Kritiski analizējiet sniegtos dokumentus: Kāda informācija trūkst, kas būtu noderīga lēmumu pieņemšanai? Kādi ir ierobežojumi prezentētajos datos? Kādi papildu pētījumi vai dati būtu nepieciešami temata labākai izpratnei?",
    "LT": "Kritiškai išanalizuokite pateiktus dokumentus: Kokios informacijos trūksta, kuri būtų naudinga priimant sprendimus? Kokie yra pateiktų duomenų apribojimai? Kokie papildomi tyrimai ar duomenys būtų reikalingi geresniam temos supratimui?",
    "MT": "Analizza b'mod kritiku d-dokumenti pprovduti: X'inhi l-informazzjoni nieqsa li tkun utli għall-ġieda tad-deċiżjonijiet? X'inhu l-limitazzjonijiet fid-dejta ppreżentata? X'inhu r-riċerka addizzjonali jew id-dejta li tkun meħtieġa għal fehim aħjar tas-suggett?",
    "PL": "Krytycznie przeanalizuj dostarczone dokumenty: Jakich informacji brakuje, które byłyby przydatne do podejmowania decyzji? Jakie są ograniczenia w przedstawionych danych? Jakie dodatkowe badania lub dane byłyby potrzebne do lepszego zrozumienia tematu?",
    "PT": "Analise criticamente os documentos fornecidos: Que informações estão faltando que seriam úteis para a tomada de decisões? Quais são as limitações nos dados apresentados? Que pesquisas ou dados adicionais seriam necessários para uma melhor compreensão do tema?",
    "RO": "Analizați critic documentele furnizate: Ce informații lipsesc care ar fi utile pentru luarea deciziilor? Care sunt limitările în datele prezentate? Ce cercetări sau date suplimentare ar fi necesare pentru o mai bună înțelegere a subiectului?",
    "SK": "Kriticky analyzujte poskytnuté dokumenty: Aké informácie chýbajú, ktoré by boli užitočné pre rozhodovanie? Aké sú obmedzenia v prezentovaných údajoch? Aké ďalšie výskumy alebo údaje by boli potrebné pre lepšie porozumenie témy?",
    "SL": "Kritično analizirajte predložene dokumente: Katere informacije manjkajo, ki bi bile uporabne za odločanje? Kakšne so omejitve v predstavljenih podatkih? Kakšne dodatne raziskave ali podatki bi bili potrebni za boljše razumevanje teme?",
    "ES": "Analice críticamente los documentos proporcionados: ¿Qué información falta que sería útil para la toma de decisiones? ¿Cuáles son las limitaciones en los datos presentados? ¿Qué investigaciones o datos adicionales serían necesarios para una mejor comprensión del tema?",
    "SV": "Analysera de tillhandahållna dokumenten kritiskt: Vilken information saknas som skulle vara användbar för beslutsfattande? Vilka är begränsningarna i de presenterade uppgifterna? Vilken ytterligare forskning eller vilka data skulle behövas för en bättre förståelse av ämnet?",
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
