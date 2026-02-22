"""
EU-FarmBook Context-Based Evaluation - 24 EU Languages
Questions designed to test RAG capabilities with search result context
Real agriculture questions that farmers/researchers ask chatbots

Structure: 
- Questions translated to all 24 EU languages
- Context (search results) kept in English only
"""

# =============================================================================
# Q2 CONTEXT - English only (5 search results for soil health)
# =============================================================================

Q2_CONTEXT_EN = [
    {
        "title": "Manure Management Tools for Efficient Fertilisation in Catalonia",
        "subtitle": "Innovative tools for sustainable manure and fertilisation management in Catalan agriculture",
        "description": "Optimised manure management tools developed under the Rural Development Programme 2014-2020 in Catalonia, Spain, improve fertilisation efficiency and reduce environmental impact. The project introduced traceability systems, GPS-enabled slurry trucks, conductivity meters, and precision application techniques like trailing shoes and band spreading. It promoted slurry acidification and use of structuring materials such as cereal straw to cut ammonia and greenhouse gas emissions.",
        "keywords": ["manure management", "fertilization", "sustainability", "Best Available Technologies", "agri-cooperative", "slurry management", "nutrient valorisation", "EIP-AGRI", "traceability system", "Catalonia"],
        "ko_content_flat": ["The manure management tools project developed innovative tools for optimising manure management and crop fertilisation, focusing on economic and environmental sustainability. Key goals included standardising manure and fertilisation management, valorising manure as a nutrient source, reducing environmental impact, and adapting technological tools to farmers needs. The project developed a traceability system for slurry transport tracking routes from livestock to crop farms, conductivity meters to measure nutrient content, and GPS technology integrated into slurry trucks. Strategies such as slurry acidification and addition of structuring materials like cereal straw were tested and found to reduce greenhouse gas and ammonia emissions cost-effectively. The project developed tools including trailing shoes and band spreading for precise application. The outcomes support improved crop productivity, environmental protection, and economic returns for farmers through better fertilisation and crop management. The results highlight that effective manure management combined with good agricultural practices enhances soil health, water retention and crop quality while reducing environmental impact."]
    },
    {
        "title": "Soil Quality and Yield in the Bulb Region: Economic Analysis for Sustainable Arable Farming",
        "subtitle": "Supporting sustainable bulb farming through soil health and economic modelling in the Netherlands",
        "description": "This poster translates improved soil quality into economic value for the Dutch Duin- and Bollenstreek region. It presents a financial model assessing the yield increase or cost savings needed to make soil-improving practices financially viable. Key measures—green manures, compost, reduced tillage, and catch crops—enhance biological, physical, and chemical soil health.",
        "keywords": ["nature-inclusive agriculture", "soil quality", "yield improvement", "cost savings", "financial model", "bulb farming", "triticale", "catch crops", "soil management", "payback period"],
        "ko_content_flat": ["This resource addresses the challenge of translating improved soil quality into measurable economic benefits for the bulb-growing sector. The project developed a financial model that calculates the required yield increase or cost savings needed to make soil-improving investments financially viable. The poster presents four key measures to improve soil quality: green manures and crop rotation, application of compost and manure, reduced tillage intensity, and inclusion of catch crops in crop plans. These measures support biological, physical and chemical soil health simultaneously. The model enables farmers to assess whether the projected yield increases from soil improvements are realistic for their specific operations."]
    },
    {
        "title": "Anaerobic Soil Disinfestation (ASD) – Practical guide",
        "subtitle": "Promotes sustainable soil management through anaerobic soil disinfestation for pathogen and pest control",
        "description": "Anaerobic soil disinfestation (ASD) is a sustainable alternative to chemical soil fumigation, effective against soil-borne pathogens, pests, and weeds. It involves incorporating 40 tons per hectare of easily degradable organic matter into the topsoil (0–40 cm), followed by watering to field capacity and covering with an airtight plastic film. This creates an anaerobic environment that triggers microbial fermentation, producing toxic volatile fatty acids that eliminate harmful organisms while beneficial microbes typically survive.",
        "keywords": ["Anaerobic Soil Disinfestation", "sustainable soil management", "soil health", "organic soil treatment", "chemical fumigant alternative", "Best4Soil", "soil pathogens", "microbial fermentation", "field capacity", "impermeable film"],
        "ko_content_flat": ["ASD is an alternative to chemical soil treatments that effectively controls soil-borne pathogens, pests and weeds without sterilising the soil. The method involves incorporating fresh easily degradable organic material into the soil at a rate of 40 tons per hectare for a 40 cm soil depth. The soil is then watered to field capacity and covered with a virtually impermeable film to create an airtight anaerobic environment. This anaerobic condition maintained for 6-8 weeks leads to microbial fermentation producing toxic volatile fatty acids that kill pathogens and pests. Beneficial soil micro-organisms generally survive preserving soil health. The approach supports sustainable agriculture by reducing reliance on chemical fumigants while maintaining soil biological function."]
    },
    {
        "title": "Solarisation and Biosolarisation for Soil Health",
        "subtitle": "Soil solarisation and biosolarisation for sustainable soil health in Mediterranean greenhouse systems",
        "description": "Solarisation disinfects soil by covering moist soil with transparent plastic for 4–6 weeks during peak summer radiation, raising temperatures to 45–55°C and inactivating fungi, nematodes, bacteria, insects, and weeds. Biosolarisation boosts efficacy by adding organic matter (C/N ratio 8–20), triggering anaerobic decomposition and biocidal compound production. The method improves microbial resilience and long-term pathogen suppression.",
        "keywords": ["biosolarisation", "soil solarisation", "soil disinfection", "Best4Soil", "Horizon 2020", "pathogen control", "organic matter incorporation", "microbial resilience", "CN ratio", "greenhouse agriculture"],
        "ko_content_flat": ["Soil solarisation is a method used to disinfect soil by covering moistened soil with transparent polyethylene film for 4-6 weeks during periods of high solar radiation. This technique is applied to control soilborne pests such as fungi, nematodes, bacteria, insects and weeds and to restore soil health. Biosolarisation enhances efficacy by incorporating fresh organic matter with a C/N ratio of 8-20 before solarisation. This triggers rapid microbial decomposition under anaerobic conditions producing biocidal compounds and shifting microbial communities. Key outcomes include improved soil health, reduced pathogen load and enhanced microbial resilience particularly when biosolarisation is applied."]
    },
    {
        "title": "Grazing Management for Long-Term Pasture Productivity",
        "subtitle": "Sustainable grassland management in Brittany and Pays de la Loire: extending sward longevity through grazing and species diversity",
        "description": "This project investigates the long-term sustainability and productivity of sown grassland-lucerne associations in western France. Results show an average herbage yield of 7 t DM/ha/year, with fescue and ryegrass dominant. Less persistent species declined rapidly, while biodiversity naturally increased over time. Summer grazing with a 73-day return interval and careful winter grazing do not degrade swards. The grassland provides 145 g metabolisable protein per kg DM, reducing nitrogen needs. Soil depth, summer drought, and grazing strategy were critical for resilience.",
        "keywords": ["grazing management", "grassland longevity", "sward composition", "fescue", "ryegrass", "lucerne", "herbage yield", "soil depth", "climate resilience", "sustainable farming"],
        "ko_content_flat": ["This project investigated the long-term sustainability and productivity of grassland systems focusing on sown grassland-lucerne associations. Key objectives included identifying factors influencing grassland longevity and assessing the impact of grazing management on sward composition and yield. Results showed an average herbage yield of 7 t DM/ha/year with significant variation linked to climate soil depth and management practices. Swards naturally diversified with increased biodiversity and a shift towards more resilient spontaneous species. The project found that well-managed swards can maintain high productivity and quality for over a decade especially when diverse species and sound grazing practices are used. The findings support the development of more sustainable low-input farming systems."]
    }
]

# =============================================================================
# Q5 CONTEXT - English only (5 search results for IPM/pest control)
# =============================================================================

Q5_CONTEXT_EN = [
    {
        "title": "Pheromone Traps for Corn Pest Monitoring and Management",
        "subtitle": "Real-time monitoring of key corn pests using pheromone traps in sustainable agriculture",
        "description": "Cap2020's Cap Trap Creep uses pheromone-based monitoring to detect and track key corn pests—corn seedling maggot, European corn borer, and corn earworm—in real time. Designed for integrated pest management (IPM), the system supports data-driven decisions in open-field and protected vegetable systems. Part of the Horizon 2020-funded Smart Protect network, the tool enables early infestation detection, severity assessment, and targeted control.",
        "keywords": ["Cap2020", "CapTrap Creep", "pheromone trap", "corn pests", "integrated pest management", "monitoring", "Atherigona oryzae", "Ostrinia nubilalis", "Helicoverpa armigera", "Horizon 2020"],
        "ko_content_flat": ["This document describes the use of pheromone-based traps for monitoring insect pests in corn production focusing on the Cap Trap Creep developed by Cap2020. The tool is designed for real-time tracking of key corn pests including the corn seedling maggot, European corn borer and corn earworm. The method supports integrated pest management by enabling farmers and advisors to detect pest presence, assess infestation levels and select appropriate control measures. The project is part of the Smart Protect thematic network funded under Horizon 2020 which promotes cross-regional knowledge sharing on smart IPM solutions. Field testing over several years confirmed the trap's effectiveness in mass trapping and monitoring. The approach enhances decision-making in pest control through timely data-driven interventions."]
    },
    {
        "title": "Modelling Western Corn Rootworm Risk on Austrian Cropland",
        "subtitle": "Crop rotation limits and climate change impact on Western Corn Rootworm risk in Austrian agriculture",
        "description": "This study assesses Western Corn Rootworm infestation risk on Austrian cropland using an integrated land use model. It evaluates the impact of maize rotation limits (10%, 25%, 50%) and climate change on pest abundance and economic returns. Results show that restricting maize to 10% of rotations reduces WCR abundance by 99.9% compared to unrestricted rotations. However, such restrictions reduce net returns and increase economic variability. The findings highlight trade-offs between pest control, climate adaptation, and agricultural profitability.",
        "keywords": ["Western Corn Rootworm", "crop rotation", "climate change", "infestation risk", "economic impact", "maize", "pest management", "Austria", "COMBIRISK", "WCR"],
        "ko_content_flat": ["This study examines the risk of Western Corn Rootworm infestation on Austrian cropland. The research investigates the impact of crop rotation regulations specifically upper limits on maize share in rotations and climate change on WCR infestation risk. Key findings indicate that restricting maize to 10% of crop rotations reduces WCR abundance by 99.9% compared to unrestricted rotations, with the highest reduction under dry conditions. However such restrictions lead to declining net returns especially under dry scenarios. The study concludes that crop rotation regulations can effectively reduce WCR pressure but must be regionally tailored. Robust farm and region-specific assessments are essential for balancing pest management, climate adaptation and agricultural profitability."]
    },
    {
        "title": "Chemical-Free Maize Cultivation with Underground Strip Plowing",
        "subtitle": "Chemical-free maize cultivation using mechanical grass suppression on sandy soil: yield, soil quality, and biodiversity outcomes",
        "description": "This two-year study on sandy soil in the Netherlands evaluates chemical-free maize cultivation using minimal soil disturbance in a living grass sward. It compares mechanical, electrical, and mulching grass suppression methods against glyphosate control. Strip mulching, especially in-row, shows strong potential for sustainable maize production, though effective early-season grass control remains a challenge. The findings support advancing mechanical suppression techniques to replace herbicides.",
        "keywords": ["maize", "chemical-free farming", "mechanical weed control", "strip mulching", "sustainable agriculture", "soil quality", "biodiversity", "grass suppression", "POP3", "Netherlands"],
        "ko_content_flat": ["This two-year study investigated chemical-free maize cultivation on sandy soil using minimal soil disturbance in a living grass sward. The research compared various grass suppression techniques including chemical treatment with glyphosate, mechanical mowing, electrocution using the Zasso machine and mulching. Results show that strip mulching particularly the in-row variant shows strong potential for chemical-free maize production, improving biodiversity and reducing chemical use. Glyphosate residues and its breakdown product were detected in soil at significant levels four months after application, highlighting environmental risks. The results support further development of mechanical grass suppression methods to achieve high yields without herbicides."]
    },
    {
        "title": "Barn Owls for Rodent Control in Agriculture",
        "subtitle": "Barn owls combat rodent infestation through nature-based pest control",
        "description": "The community of Deneia in Cyprus implemented a nature-based solution using barn owls to control rat and mouse populations, reducing dependence on chemical pesticides and promoting organic farming. The project installed artificial nesting boxes and used motion-sensor cameras to monitor activity. Results were significant: rodent populations declined markedly, allowing 70% of local farmers to adopt organic farming. The project draws on Israel's long-standing barn owl programme which reduced rodenticide use by 80% since 2006.",
        "keywords": ["barn owl", "rodent control", "biological pest control", "organic farming", "nature-based solution", "integrated pest management", "Cyprus", "artificial nest box"],
        "ko_content_flat": ["This project implemented a nature-based solution using barn owls for rodent control in agricultural areas. Barn owls consume up to 1000 rodents annually with 96% of their diet consisting of mice, rats and shrews. Artificial nesting boxes were constructed and installed in strategic locations near rodent habitats. Results were significant: rodent populations declined markedly allowing farmers to adopt organic farming. The project reduced reliance on chemical pesticides and enhanced ecosystem resilience. This demonstrates the viability of nature-based solutions for integrated pest management in agriculture."]
    },
    {
        "title": "Intercropping and Pest Risk Management in Crop Systems",
        "subtitle": "Intercropping strategies for pest risk reduction in agricultural cultivation",
        "description": "This study examines intercropping's impact on pest risk in cultivation systems. Key factors for successful intercropping include diverse species-rich mixtures, high sowing density, and appropriate sowing timing. Field monitoring is critical during the emergence period, especially under dry warm conditions. Effective pest management requires understanding pest thresholds, biological control options, and targeted insecticide applications. A warning service provides real-time pest monitoring data across regions to support IPM decision-making.",
        "keywords": ["intercropping", "pest risk", "integrated pest management", "biological control", "insecticide", "field monitoring", "pest thresholds", "warning service", "crop protection"],
        "ko_content_flat": ["This study investigated intercropping strategies for pest risk management. Key factors for successful intercropping include diverse species-rich mixtures, high sowing density and appropriate sowing timing. Field monitoring is critical during crop emergence, especially under dry warm conditions. Pest thresholds are defined for different growth stages to guide intervention decisions. Biological control is often sufficient for many pests. Chemical control using targeted insecticides is effective when combined with proper timing. A free warning service provides real-time pest monitoring data across regions to support integrated pest management. Targeted applications can successfully manage moderate infestations while minimising environmental impact."]
    }
]

# =============================================================================
# Q4 CONTEXT - English only (5 search results for EU funding)
# =============================================================================

Q4_CONTEXT_EN = [
    {
        "title": "Grazing Management for Small Ruminants",
        "subtitle": "Sustainable grazing solutions for small ruminants in challenging Austrian pastures",
        "description": "This project developed sustainable grazing solutions for small ruminants in Austria, addressing challenges on steep, dry, or poorly drained pastures. Funded by the EIP-AGRI programme, it tested innovative seed mixtures with anti-parasitic properties and grazing systems. Results showed Short-Rye Pasture improved weight gain and reduced parasite load on well-drained soils. The project was published in 2025 by bio austria Bundesverband with support from the Austrian Federal Ministry of Agriculture and EAFRD.",
        "keywords": ["grazing", "small ruminants", "organic farming", "Austria", "EU Organic Regulation", "sheep", "goats", "short-rye pasture", "parasite load", "EIP-AGRI"],
        "ko_content_flat": ["The EIP-Weideinnovationen project funded under the European Innovation Partnership for Agricultural Sustainability aimed to develop innovative practical and sustainable solutions for small ruminant grazing under challenging conditions in Austria. The project ran from 2021 to 2023 and involved six organic farming enterprises and research institutions. Key objectives included testing site-specific seed mixtures with anti-parasitic properties, evaluating novel grazing systems, and assessing impact on parasite load and animal performance. Results showed that short-rye pasture systems led to higher daily weight gains and lower parasite loads. The project was supported by the Austrian Federal Ministry of Agriculture and the European Agricultural Fund for Rural Development EAFRD under the EIP-AGRI programme."]
    },
    {
        "title": "Empowerment of Women Farmers Through Olive Oil and Renewable Energy",
        "subtitle": "Empowering rural women through energy innovation and social impact assessment",
        "description": "A case study on the FARMWELL project, funded by the EU Horizon 2020 programme, assessing the social return on investment of the Women in Olive Oil initiative. The study evaluates long-term wellbeing impacts of 120 female farmers across five years. Data revealed a total benefit value of €890,651, with a benefit-investment ratio of 3.64:1. The largest social benefit came from improved trust and belonging within the cooperative. The study highlights the role of social innovation in advancing gender equity and rural economic resilience.",
        "keywords": ["Women in Olive Oil", "Social Return on Investment", "Energy Communities", "Social Innovation", "Wellbeing", "FARMWELL project", "Horizon 2020", "Rural Empowerment", "Gender Equity", "Community Cohesion"],
        "ko_content_flat": ["The FARMWELL project funded by the European Union's Horizon 2020 programme explores the social return on investment of the Women in Olive Oil initiative. The project assesses the long-term wellbeing impacts of 120 members over a five-year period. The total benefit value is €890,651 with a benefit-investment ratio of 3.64:1. The largest social benefit comes from an improved sense of trust and belonging among cooperative members. Economic wellbeing accounts for 20% of total value, driven by improved leadership and entrepreneurial skills. The project demonstrates that social innovation significantly enhances gender equity, community cohesion and economic resilience among rural women."]
    },
    {
        "title": "CaVin: Hydrodynamic Cavitation for Grape Pomace Energy Valorisation",
        "subtitle": "Enhancing methane yield from wine industry by-products via hydrodynamic cavitation pre-treatment",
        "description": "The CaVin project enhances biogas production from grape pomace using hydrodynamic cavitation pre-treatment. Conducted by CRPA in Emilia-Romagna, it increased methane yield by 83% in batch tests and 47% in continuous flow reactors. The method improves digestibility of lignocellulosic biomass, reduces particle size, and enhances reactor stability. Supported by EIP-AGRI and the EU Rural Development Programme, the project promotes circular economy in wine production.",
        "keywords": ["cavitation", "grape pomace", "biogas", "pre-treatment", "anaerobic digestion", "renewable energy", "CRPA", "Emilia-Romagna", "EIP-AGRI"],
        "ko_content_flat": ["The CaVin project implemented under the Emilia-Romagna Regional Programme for Rural Development focused on enhancing the energy valorisation of grape pomace through controlled hydrodynamic cavitation pre-treatment. Grape pomace rich in cellulose, hemicellulose and lignin is a recalcitrant biomass that poses challenges in anaerobic digestion. The project tested hydrodynamic cavitation as a pre-treatment method to improve digestibility. Results showed an 83% increase in methane yield in batch tests and 47% in continuous tests. The project was funded under the European Innovation Partnership for Agriculture and Food EIP-AGRI."]
    },
    {
        "title": "Suckling Calf Rearing with Pasture Access",
        "subtitle": "Suckling calf rearing with pasture access in organic dairy farming: practical insights from 71 European farms",
        "description": "This brochure presents practical insights from 71 farms in Austria, Germany, and Switzerland on suckling calf rearing with pasture access in organic dairy systems. It addresses challenges and benefits under the EU Organic Regulation (2018/848). Key systems include full-time, half-day, and short-term maternal contact. Benefits include improved calf vitality, reduced stress, and lower labour needs. Economic analysis shows full-time systems become cost-effective when labour and infrastructure savings are considered. Funded by the EU Horizon Europe framework.",
        "keywords": ["calf rearing", "pasture access", "organic farming", "dairy", "suckling systems", "animal welfare", "grazing innovations", "EIP project", "European Union", "Horizon Europe"],
        "ko_content_flat": ["This brochure produced within the EIP Innovation Partnerships project Weideinnovationen in Austria focuses on suckling calf rearing with pasture access in organic dairy farming. Funded by the European Union and part of the Horizon Europe framework, the project addresses new requirements under the EU Organic Regulation 2018/848. Based on data from 71 farms across Austria, Germany and Switzerland, it covers suckling calf rearing systems and economic viability. Benefits reported include better calf vitality, reduced parasite loads, improved social development and reduced need for manual feeding. The project was supported by the Austrian Ministry of Agriculture and EAFRD."]
    },
    {
        "title": "Protected Lettuce: N-P-K vs N-only Fertilisation in Winter Tunnel Cultivation",
        "subtitle": "Winter lettuce trial in plastic tunnel comparing complete and low-input fertilisation in southern France",
        "description": "This study evaluates two fertilisation strategies—complete N-P-K versus N-only—on winter lettuce in a plastic tunnel at Rognonas, France. Despite high P and K inputs in the complete fertiliser, no significant differences were found in yield. Soil analysis revealed high baseline nutrients exceeding critical thresholds. The results support the REVEIL approach—omitting P and K fertilisation—on this site. Funded by the EU's EAFRD under EIP-AGRI.",
        "keywords": ["lettuce", "phosphorus", "potassium", "fertilisation", "yield", "nutrient analysis", "REVEIL project", "winter cultivation", "soil fertility", "N-P-K", "low-input farming", "France"],
        "ko_content_flat": ["This experiment conducted within the REVEIL project assessed the impact of two fertilisation strategies on winter lettuce cultivation in a plastic tunnel in France. The trial compared complete fertilisation N-P-K versus a low-input approach with no phosphorus or potassium inputs. Despite a 115 kg/ha K and 92 kg/ha P input in the complete fertiliser, no measurable difference was detected. The results suggest that existing soil nutrient reserves were sufficient. The study concludes that the REVEIL approach omitting P and K fertilisation may be viable. The project is supported by the European Union through EAFRD."]
    }
]

# =============================================================================
# Q3 CONTEXT - English only (5 search results for climate adaptation)
# =============================================================================

Q3_CONTEXT_EN = [
    {
        "title": "Deep Root Training Tubes for Drought-Resistant Hedgerows in Cyprus",
        "subtitle": "Enhancing climate resilience in Cyprus through native drought-resistant planting protocols and community-led reforestation",
        "description": "The EU-funded LIFE-AGROASSIS project developed an innovative planting protocol using Deep Root Training Tubes (DRTT) to enhance climate resilience in Cyprus's arid and semi-arid regions. The system promotes deep root development in drought-resistant native species through 90-day nursery training, reducing post-planting irrigation. Community volunteers from 20 organisations are trained to plant 4,000 deep-rooted plants annually. Results will inform policy integration into the Common Agricultural Policy.",
        "keywords": ["drought-resistant plants", "deep root training tubes", "desertification control", "native Cypriot species", "soil rehabilitation", "climate resilience", "LIFE-AGROASSIS project", "sustainable agriculture", "carbon sequestration", "pollinator support"],
        "ko_content_flat": ["The LIFE-AGROASSIS project focuses on enhancing climate resilience in arid and semi-arid agricultural regions of Cyprus prone to desertification. The project developed an innovative planting protocol using Deep Root Training Tubes (DRTT) to grow drought-resistant seedlings in nurseries. The DRTT system promotes deep root development over 90 days, preparing plants for arid conditions and reducing post-planting irrigation needs. 18 native Cypriot plant species are in active propagation including trees, shrubs and herbs selected for their ability to establish deep root systems, improve degraded soils, support pollinators and enhance ecosystem services. Community volunteers from up to 20 organisations are trained to support planting events. Long-term benefits include improved soil quality, microclimate regulation, carbon sequestration and potential income from carbon farming."]
    },
    {
        "title": "Paulownia for Agroforestry and Carbon Sequestration in the Mediterranean",
        "subtitle": "Fast-growing Paulownia trees combat desertification and sequester carbon in Tunisia and Portugal",
        "description": "Paulownia fortunei and Paulownia elongata show strong potential for agroforestry and carbon sequestration in Tunisia and Portugal, thriving in poor, sandy soils with low water needs. In Tunisia, 3.5 hectares of 6-month-old plantations yield significant CO2 sequestration (30–40 tonnes/ha/year). The trees enhance water retention, reduce erosion, and support apiculture. In Portugal, Paulownia provides fast-growing timber for furniture and construction, with 7–10 year harvest cycles. Its deep roots stabilise degraded lands while canopy cover boosts crop yields by 15–30% in agroforestry systems.",
        "keywords": ["Paulownia", "agroforestry", "carbon sequestration", "drought resilience", "soil rehabilitation", "sustainable forestry", "honey production", "Mediterranean", "RESALLIANCE", "climate change mitigation"],
        "ko_content_flat": ["Paulownia is being explored for its potential in agroforestry systems and carbon sequestration in Tunisia and Portugal. The species are promoted for their resilience to drought, suitability for poor and sandy soils and low management requirements. Research indicates Paulownia can sequester 30-40 tonnes of CO2 per hectare annually. The species also supports apiculture with its spring flowering providing a rich nectar source. In Portugal the tree contributes to sustainable forestry by offering a renewable source of high-quality timber with harvest cycles of 7-10 years. The trees deep root systems stabilise slopes and degraded lands while their canopy provides microclimate benefits which can increase crop yields by 15-30% in agroforestry systems."]
    },
    {
        "title": "Vineyards as Natural Firebreaks: Forest Fire Prevention in Cyprus",
        "subtitle": "Vineyards as natural firebreaks in Cyprus: A case study in climate resilience and sustainable land management",
        "description": "The Tsiakkas Winery in Agros, Cyprus, demonstrates how vineyards act as natural firebreaks during forest fires. In 2007 and 2024, surrounding vineyards containing native varieties like Xynisteri, Mavro, and Commandaria protected the winery and internal pine trees from intense flames. Deep roots, high leaf moisture, and island-like planting patterns reduce fire intensity and support firefighting. Recognised by the Cypriot Department of Forests, this nature-based solution enhances fire resilience, prevents soil erosion, and delivers economic and environmental co-benefits.",
        "keywords": ["vineyards", "firebreak", "soil management", "forest fire prevention", "natural barriers", "sustainable agriculture", "climate resilience", "Mediterranean landscape", "European projects"],
        "ko_content_flat": ["The Tsiakkas winery serves as a case study demonstrating the effectiveness of vineyards as natural firebreaks in preventing the spread of forest fires. In 2007 during a major wildfire the vineyards surrounding the winery acted as a firebreak protecting the winery and pine trees. The success relies on deep root systems that access soil moisture, high leaf moisture content and strategic island-like planting patterns that create physical barriers. The Department of Forests recognises vineyards as a valuable tool in fire prevention. Vineyards also deliver multiple co-benefits: soil erosion control, economic returns from wine production and landscape aesthetics. The approach is particularly suited to dry hot climates and degraded or abandoned lands."]
    },
    {
        "title": "Restoration of Degraded Mountain Catchments for Flood Mitigation",
        "subtitle": "Mountain catchment restoration in Pindos, Greece, 1955–1980s: A model for climate-resilient flood mitigation",
        "description": "The Greek Forest Service restored the Metsovitikos torrent catchment in Pindos, Greece, from 1955 to the 1980s, implementing phytotechnical measures including 1,000 check-dams and 70 gabion walls, reforesting over 500 hectares with 1.8 million seedlings. This large-scale project significantly reduced soil erosion, improved water retention, and mitigated downstream flooding. A 2004 study confirmed its effectiveness. The project is now a model for nature-based flood resilience, highlighting the need for renewed investment to address climate-driven risks and expanding wildfire threats in the Mediterranean.",
        "keywords": ["watershed restoration", "flooding mitigation", "soil erosion prevention", "phytotechnical works", "mountain catchment restoration", "Metsovitikos torrent", "Greek Forest Service", "check-dams", "gabion walls", "reforestation", "climate resilience", "nature-based solutions"],
        "ko_content_flat": ["The Greek Forest Service implemented a comprehensive mountain catchment restoration project in the Metsovitikos torrent watershed between 1955 and the 1980s. The project involved constructing over 1000 check-dams and 70 gabion walls followed by reforestation with 1.8 million seedlings across more than 500 hectares. By the 1980s results were evident: vegetation cover increased, water retention improved, peak flows decreased and downstream flooding was substantially mitigated. Recent extreme weather events linked to climate change have highlighted the urgent need to revive such mountain-based restoration. The project is cited as a model for sustainable low-tech nature-based flood mitigation."]
    },
    {
        "title": "Community Pastoral System of the Agdal in Morocco for Socio-Ecological Resilience",
        "subtitle": "Community-based seasonal pasture management for ecological resilience and climate adaptation in the High Atlas, Morocco",
        "description": "The Agdal system in Morocco's High Atlas is a community-based natural resource management model that ensures sustainable pasture use through seasonal closures, promoting ecological regeneration, biodiversity, and social cohesion. Rooted in customary governance, it involves local tribes in decision-making via jmaa assemblies. By rotating grazing rights and protecting vegetation during critical growth periods, Agdals maintain soil fertility, support livestock during droughts, and enhance microclimates. The system strengthens rural livelihoods and offers a proven, locally adapted solution for climate adaptation.",
        "keywords": ["community-based natural resource management", "Agdal", "seasonal pastures", "participatory decision-making", "socio-ecological resilience", "biodiversity", "climate change", "pastoral management", "Atlas Mountains", "traditional knowledge", "climate resilience"],
        "ko_content_flat": ["The Agdal pastoral system in the High Atlas mountains of Morocco is a community-based natural resource management system that promotes socio-ecological resilience. The system involves local communities in participatory decision-making through tribal assemblies which establish rules for sustainable use. Agdals are seasonal pasture reserves closed to grazing during critical plant growth periods to allow vegetation to regenerate. This practice supports mosaics of ecological habitats enhancing plant diversity. The system delivers multiple benefits: equitable access to resources, strengthened social cohesion and support for livestock during droughts. The Agdal model exemplifies a biocultural conservation system that integrates traditional ecological knowledge, participatory governance and sustainable land use."]
    }
]

# =============================================================================
# Q1 CONTEXT - English only (5 search results from API)
# =============================================================================

Q1_CONTEXT_EN = [
    {
        "title": "Innovating Together for Emission-Free Weed Control: Groesbeek Field Demonstration Results 2023",
        "subtitle": "Reducing herbicide use in sugar beets and maize through integrated mechanical and chemical weed control in Groesbeek, 2023",
        "description": "Reduced crop protection product use in sugar beets and biological maize through integrated mechanical and chemical weed control in Groesbeek, 2023. Field trials evaluated four strategies in sugar beets: full chemical, integrated row spraying with mechanical weeding, fully mechanical control, and Conviso One herbicide. In maize, early hoeing and camera-guided Garford Robocrop InterRow with finger weeder attachments were tested. Weed pressure was lowest in fully mechanical sugar beet plots (5% cover on 14 June), and Conviso One reduced active ingredients by 95% (0.195 kg/ha vs. 3.89 kg/ha).",
        "keywords": ["emission reduction", "crop protection", "mechanical control", "weed control", "sugar beets", "maize", "organic farming", "Conviso One", "Garford Robocrop", "inter-row hoeing"],
        "ko_content_flat": ["This project, part of the POP3 initiative and co-funded by the Dutch government, aimed to reduce the use of crop protection products in sugar beets and biological maize through integrated mechanical and chemical weed control. Field demonstrations took place in Groesbeek, Gelderland, in 2023. The study evaluated four strategies in sugar beets: a) full chemical control using a traditional low-drift sprayer (LDS) system; b) integrated control with row spraying and mechanical weeding between rows; c) fully mechanical control with inter-row hoeing; and d) full chemical control using Conviso One herbicide. In biological maize, mechanical weeding was tested on loess soil using seedbed preparation followed by hoeing and later use of a camera-guided Garford Robocrop InterRow machine with finger weeder attachments. Results show that mechanical weeding is most effective on small weeds, and early inter-row cultivation improves soil structure."]
    },
    {
        "title": "AGROSUS: Sustainable Weed Management in European Farming",
        "subtitle": "Agroecological weed management strategies for sustainable farming across Europe",
        "description": "The AGROSUS project, funded by Horizon Europe (grant agreement No GA 101084084), develops co-created agroecological weed management strategies for 31 key crops across 11 EU biogeographic regions. Running from June 2023 to May 2027, it involves 16 partners from 11 countries. The project targets sustainable, fair, and safe weed control in conventional, organic, and mixed farming systems through 14 Regional Stakeholder Communities and 24 Crop-Linked Groups.",
        "keywords": ["Agroecology", "weed management", "agroecological strategies", "synthetic herbicides", "Horizon Europe", "Farm to Fork Strategy", "sustainable agriculture", "stakeholder co-creation"],
        "ko_content_flat": ["The AGROSUS project, coordinated by the Universidad de Vigo (UVigo) in Spain, is a Horizon Europe-funded initiative (grant agreement No GA 101084084) running from June 2023 to May 2027 (48 months) with a total budget of €4,999,863.75. The project involves 16 partners from 11 European and associated countries. Its primary goal is to develop and co-create agroecological strategies for sustainable, fair, and safe weed management across conventional, organic, and mixed farming systems in all 11 EU biogeographic regions. The project focuses on 31 key crops including wheat, maize, potatoes, apples, olives, almonds, and tomatoes."]
    },
    {
        "title": "Rat Population Control Using Barn Owls in Deneia",
        "subtitle": "Barn owls combat rodent infestation in Cyprus' Deneia community through nature-based pest control",
        "description": "Resettlement of farmers on abandoned lands in Deneia, Cyprus, with implementation of a nature-based solution using barn owls to control rat and mouse populations, reducing dependence on chemical pesticides and promoting organic farming. The project, co-funded by the EU under the RESALLIANCE initiative, led to marked reduction in pests through installation of artificial nests, infrared camera monitoring, and strong community engagement including workshops, school competitions, and participatory nest construction.",
        "keywords": ["rats", "barn owl", "rodent control", "artificial nest box", "organic farming", "Deneia", "Cyprus", "nature-based solution", "wildlife conservation"],
        "ko_content_flat": ["The community of Deneia, located west of Nicosia, the capital of Cyprus, lies within and near the United Nations buffer zone, a region marked by decades of agricultural abandonment. Over the past 15 years the community faced a severe rodent infestation which hindered agricultural revival. Chemical pest control methods proved ineffective and caused environmental harm, prompting a shift to a nature-based solution. The local authority partnered with barn owls (Tyto alba), a widespread nocturnal raptor known for its high rodent predation efficiency. These owls consume up to 1000 rodents annually, with 96% of their diet consisting of mice, rats, and shrews."]
    },
    {
        "title": "Chemical-Free Maize Cultivation with Underground Strip Plowing in Grassland",
        "subtitle": "Chemical-free maize cultivation using mechanical grass suppression on sandy soil: yield, soil quality, and biodiversity outcomes",
        "description": "This two-year study on sandy soil in the Netherlands, funded by the Province of Drenthe and the EU (POP3), evaluates chemical-free maize cultivation using minimal soil disturbance (OSP) in a living grass sward. It compares mechanical, electrical, and mulching grass suppression methods - mowing, Zasso electrocution, and strip or volveld mulching - against glyphosate control. In 2019, low yields were observed due to drought, with nitrogen addition improving results. In 2020, highest yields (15 t DS/ha) were achieved with glyphosate and in-row strip mulching at location 1.",
        "keywords": ["Drenthe", "maize", "OSP", "grass suppression", "soil quality", "biodiversity", "yield", "mulching", "Zasso", "glyphosate", "mineral nitrogen", "agrobiodiversity", "chemical-free farming", "sandy soil", "living mulch", "POP3"],
        "ko_content_flat": ["This two-year study, funded by the Province of Drenthe and the EU (POP3), investigated chemical-free maize cultivation on sandy soil using minimal soil disturbance (underground strip ploughing 'OSP') in a living grass sward, with focus on yield, soil quality, and biodiversity. Conducted at two locations in 2019 and 2020, the research compared various grass suppression techniques - chemical treatment with glyphosate, mechanical mowing twice or four times monthly, electrocution using the Zasso machine, and mulching - against a glyphosate control. Results show that strip mulching, particularly the inter-row variant, shows strong potential for chemical-free maize production, improving biodiversity and reducing chemical use."]
    },
    {
        "title": "Image-based Selective Mechanical Weeding for Field Vegetables",
        "subtitle": "Supporting sustainable vegetable farming through advanced image-based mechanical weeding technologies in Horizon Europe's Smart Protect network",
        "description": "Mechanical weeding using image-based technologies like Garford Robocrop and Naoi-Dino supports non-chemical weed control in vegetable crops. These systems enhance precision in inter- and intra-row weeding, reducing herbicide use and soil disturbance. Part of Horizon Europe's Smart Protect network, the solutions improve decision-making for farmers and advisors in open-field and greenhouse production.",
        "keywords": ["IPM", "image-based weeding", "mechanical weeding", "vegetables", "intra-row weeding", "Garford Robocrop", "Naoi-Dino", "Horizon Europe", "sustainable farming", "smart farming"],
        "ko_content_flat": ["Smart Protect is a thematic network under the Horizon Europe programme supporting cross-regional knowledge sharing on smart integrated pest management (IPM) solutions for vegetable production in open fields and greenhouses across Europe. The initiative focuses on improving decision-making for farmers and advisors, particularly in crops including tomato, cucumber, bell pepper, lettuce, alliums, and cabbage. Image-based selective mechanical weeding is a key non-chemical weed management strategy within IPM. Two technologies exemplify this approach: Garford Robocrop and Naoi-Dino. The Garford Robocrop system is a tractor-mounted attachment designed for both inter-row and intra-row weeding. It uses a digital video camera to capture real-time images of crops ahead of the toolbar."]
    }
]

# =============================================================================
# CONTEXT DATA - All questions use English context
# =============================================================================

CONTEXT_DATA = {
    "Q1": Q1_CONTEXT_EN,  # Organic weed control for cereal crops
    "Q2": Q2_CONTEXT_EN,  # Soil health - 5 search results
    "Q3": Q3_CONTEXT_EN,  # Climate adaptation - 5 search results
    "Q4": Q4_CONTEXT_EN,  # EU funding - 5 search results
    "Q5": Q5_CONTEXT_EN,  # IPM/pest control - 5 search results
}

# =============================================================================
# QUESTIONS - All 5 questions in 24 EU languages
# =============================================================================

# Q1: Organic weed control for cereal crops
Q1_TRANSLATIONS = {
    "BG": "Какви органични методи за борба с плевели препоръчвате за зърнени култури в умерен климат? Искам алтернативи на хербицидите.",
    "HR": "Koje organske metode suzbijanja korova preporučujete za žitarice u umjerenoj klimi? Želim alternative herbicidima.",
    "CS": "Jaké organické metody hubení plevelů doporučujete pro obilniny v mírném podnebí? Chci alternativy k herbicidům.",
    "DA": "Hvilke organiske ukrudtsbekæmpelsesmetoder anbefaler du til kornafgrøder i tempereret klima? Jeg vil have alternativer til herbicider.",
    "NL": "Welke biologische onkruidbestrijdingsmethoden raadt u aan voor graangewassen in een gematigd klimaat? Ik wil alternatieven voor herbiciden.",
    "EN": "What organic weed control methods do you recommend for cereal crops in a temperate climate? I want alternatives to herbicides.",
    "ET": "Milliseid orgaanilisi umbrohutõrje meetodeid soovitate teraviljakultuuridele parasvöötmes? Tahan herbitsiidide alternatiive.",
    "FI": "Mitkä orgaaniset rikkakasvien torjuntamenetelmät suosittelet viljakasveille lauhkeassa ilmastossa? Haluan vaihtoehtoja herbicideille.",
    "FR": "Quelles méthodes de lutte biologique contre les mauvaises herbes recommandez-vous pour les céréales en climat tempéré? Je veux des alternatives aux herbicides.",
    "DE": "Welche biologischen Unkrautbekämpfungsmethoden empfehlen Sie für Getreide in gemäßigtem Klima? Ich möchte Alternativen zu Herbiziden.",
    "EL": "Ποιες βιολογικές μέθοδοι καταπολέμησης ζιζανίων συστήνετε για σιτηρά σε εύκρατο κλίμα; Θέλω εναλλακτικές λύσεις στα ζιζανιοκτόνα.",
    "HU": "Milyen szerves gyomirtási módszereket ajánl a gabonafélék számára mérsékelt éghajlaton? Herbicid alternatívákat szeretnék.",
    "GA": "Cad iad na modhanna orgánacha rialaithe fiail atá agat a mholadh do chruithneacht i ghaothafána? Teastaíonn malairtí uaim ar herbicidí.",
    "IT": "Quali metodi biologici di diserbo consigliate per le colture cerealicole in clima temperato? Voglio alternative agli erbicidi.",
    "LV": "Kādas organiskās nezāļu kontroles metodes jūs iesakāt graudaugiem mērenā klimatā? Es vēlos alternatīvas herbicīdiem.",
    "LT": "Kokias organines piktžolių kontrolės metodikas rekomenduojate javams vidutinio klimato sąlygomis? Noriu alternatyvų herbicidams.",
    "MT": "X' metodi organiċi tal-kontroll tal-ħaxix ħażin tirrakkomanda għall-qamħirrun fi klimat moderat? Irid alternattivi għall-herbicides.",
    "PL": "Jakie organiczne metody zwalczania chwastów polecasz dla zbóż w klimacie umiarkowanym? Chcę alternatywy dla herbicydów.",
    "PT": "Que métodos biológicos de controle de ervas daninhas você recomenda para culturas de cereais em clima temperado? Quero alternativas aos herbicidas.",
    "RO": "Ce metode organice de combatere a buruienilor recomandați pentru culturile de cereale în climă temperată? Vreau alternative la erbicide.",
    "SK": "Aké organické metódy kontroly burín odporúčate pre obilniny v miernom podnebí? Chcem alternatívy k herbicídom.",
    "SL": "Katere organske metode nadzora plevela priporočate za žita v zmernem podnebju? Želim alternative herbicidom.",
    "ES": "¿Qué métodos orgánicos de control de malezas recomienda para cultivos de cereales en clima templado? Quiero alternativas a los herbicidas.",
    "SV": "Vilka organiska ogräsbekämpningsmetoder rekommenderar du för spannmålsgrödor i tempererat klimat? Jag vill ha alternativ till herbicider."
}

# Q2: Soil health practices
Q2_TRANSLATIONS = {
    "BG": "Как мога да подобря здравето на почвата в моята овощна градина след години на интензивно земеделие?",
    "HR": "Kako mogu poboljšati zdravlje tla u svom voćnjaku nakon godina intenzivnog poljoprivrednog gospodarenja?",
    "CS": "Jak mohu zlepšit zdraví půdy v ovocném sadu po letech intenzivního zemědělského hospodaření?",
    "DA": "Hvordan kan jeg forbedre jordens sundhed i min frugtplantage efter år med intensivt landbrug?",
    "NL": "Hoe kan ik de bodemgezondheid in mijn boomgaard verbeteren na jaren van intensieve landbouw?",
    "EN": "How can I improve soil health in my orchard after years of intensive farming?",
    "ET": "Kuidas saan parandada mulla tervist oma puuviljaaias pärast aastaid intensiivset põllumajandust?",
    "FI": "Kuinka voin parantaa maaperän terveyttä hedelmätarhassani vuosien intensiivisen maatalouden jälkeen?",
    "FR": "Comment puis-je améliorer la santé du sol dans mon verger après des années d'agriculture intensive?",
    "DE": "Wie kann ich die Bodengesundheit in meinem Obstgarten nach Jahren intensiver Landwirtschaft verbessern?",
    "EL": "Πώς μπορώ να βελτιώσω την υγεία του εδάφους στον οπωρώνα μου μετά από χρόνια εντατικής γεωργίας;",
    "HU": "Hogyan javíthatom a talaj egészségét a gyümölcsösömben az évekig tartó intenzív mezőgazdaság után?",
    "GA": "Conas is féidir liom sláinte ithreach a fheabhsú i mo ghorta tar éis blianta de thalmhaíocht dhian?",
    "IT": "Come posso migliorare la salute del suolo nel mio frutteto dopo anni di agricoltura intensiva?",
    "LV": "Kā es varu uzlabot augsnes veselību savā dārzā pēc gadiem ilgas intensīvas lauksaimniecības?",
    "LT": "Kaip galiu pagerinti dirvožemio sveikatą savo vaisių sode po intensyvios žemdirbystės metų?",
    "MT": "Kif nista' ntejjeb is-saħħa tal-ħamrija f'ġnieni tal-frott wara snin ta' bidwi intensiv?",
    "PL": "Jak mogę poprawić zdrowie gleby w moim sadzie po latach intensywnego rolnictwa?",
    "PT": "Como posso melhorar a saúde do solo no meu pomar após anos de agricultura intensiva?",
    "RO": "Cum pot îmbunătăți sănătatea solului în livada mea după ani de agricultură intensivă?",
    "SK": "Ako môžem zlepšiť zdravie pôdy v mojom ovocnom sade po rokoch intenzívneho poľnohospodárstva?",
    "SL": "Kako lahko izboljšam zdravje tal v svojem sadovnjaku po letih intenzivnega kmetijstva?",
    "ES": "¿Cómo puedo mejorar la salud del suelo en mi huerto después de años de agricultura intensiva?",
    "SV": "Hur kan jag förbättra jordhälsan i min fruktodling efter år av intensivt jordbruk?"
}

# Q3: Climate adaptation strategies
Q3_TRANSLATIONS = {
    "BG": "Кои са най-добрите практики за адаптиране на моята ферма към променящия се климат в Средиземноморския регион?",
    "HR": "Koje su najbolje prakse prilagodbe moje farme promjenjivoj klimi u mediteranskom području?",
    "CS": "Jaké jsou nejlepší postupy pro přizpůsobení mé farmy měnícímu se klimatu ve Středomoří?",
    "DA": "Hvad er de bedste praksisser for at tilpasse min gård til det skiftende klima i Middelhavsområdet?",
    "NL": "Wat zijn de beste praktijken om mijn boerderij aan te passen aan het veranderende klimaat in het Middellandse Zeegebied?",
    "EN": "What are the best practices for adapting my farm to the changing climate in the Mediterranean region?",
    "ET": "Mis on parimad tavad oma talu kohandamiseks muutuva kliimaga Vahemere piirkonnas?",
    "FI": "Mitä ovat parhaat käytännöt maatilani sopeuttamiseksi muuttuvaan ilmastoon Välimeren alueella?",
    "FR": "Quelles sont les meilleures pratiques pour adapter mon exploitation aux changements climatiques dans la région méditerranéenne?",
    "DE": "Was sind die besten Praktiken, um meinen Betrieb an den sich wandelnden Klima im Mittelmeerraum anzupassen?",
    "EL": "Ποιες είναι οι καλύτερες πρακτικές για την προσαρμογή της φάρμας μου στο μεταβαλλόμενο κλίμα στην περιοχή της Μεσογείου;",
    "HU": "Mik a legjobb gyakorlatok a gazdaságom alkalmazkodásához a változó éghajlathoz a Földközi-tengeri régióban?",
    "GA": "Cad iad na cleachtais is fearr chun mo fheirm a oiriúnú don aeráid atá ag athrú sa réigiún Meánmhara?",
    "IT": "Quali sono le migliori pratiche per adattare la mia azienda al clima che cambia nella regione mediterranea?",
    "LV": "Kādas ir labākās prakses manas saimniecības pielāgošanai mainīgajam klimatam Vidusjūras reģionā?",
    "LT": "Kokia geriausia praktika pritaikyti mano ūkį prie besikeičiančio klimato Viduržemio jūros regione?",
    "MT": "X'inhu l-aħjar prattiki biex jaġġustaw ir-ranch tiegħi għall-klima li qed tinbidel fir-reġjun tal-Mediterran?",
    "PL": "Jakie są najlepsze praktyki dostosowania mojego gospodarstwa do zmieniającego się klimatu w regionie śródziemnomorskim?",
    "PT": "Quais são as melhores práticas para adaptar minha fazenda às mudanças climáticas na região do Mediterrâneo?",
    "RO": "Care sunt cele mai bune practici pentru adaptarea fermeii mele la schimbările climatice din regiunea mediteraneană?",
    "SK": "Aké sú najlepšie postupy prispôsobenia mojej farmy meniacemu sa podnebiu v stredomorskom regióne?",
    "SL": "Katere so najboljše prakse za prilagoditev moje kmetije spreminjajočemu se podnebju v sredozemski regiji?",
    "ES": "¿Cuáles son las mejores prácticas para adaptar mi granja al cambio climático en la región mediterránea?",
    "SV": "Vilka är de bästa praxis för att anpassa min gård till det förändrade klimatet i Medelhavsområdet?"
}

# Q4: EU funding for agroecology
Q4_TRANSLATIONS = {
    "BG": "Какви програми за финансиране на ЕС са налични за млади фермери, които искат да преминат към агроекология?",
    "HR": "Koji programi financiranja EU-a dostupni su mladim poljoprivrednicima koji žele prijeći na agroekologiju?",
    "CS": "Jaké programy financování EU jsou k dispozici pro mladé zemědělce, kteří se chtějí přejít na agroekologii?",
    "DA": "Hvilke EU-finansieringsprogrammer er tilgængelige for unge landmænd, der ønsker at skifte til agroøkologi?",
    "NL": "Welke EU-financieringsprogramma's zijn beschikbaar voor jonge boeren die willen overschakelen op agro-ecologie?",
    "EN": "What EU funding programs are available for young farmers who want to transition to agroecology?",
    "ET": "Millised EL-i rahastusprogrammid on saadavad noortele talunikele, kes soovivad üle minna agroökoloogiale?",
    "FI": "Mitä EU-rahoitusohjelmia on saatavilla nuorille viljelijöille, jotka haluavat siirtyä agroekologiaan?",
    "FR": "Quels programmes de financement de l'UE sont disponibles pour les jeunes agriculteurs qui souhaitent passer à l'agroécologie?",
    "DE": "Welche EU-Finanzierungsprogramme stehen jungen Landwirten zur Verfügung, die auf Agroökologie umsteigen wollen?",
    "EL": "Ποια προγράμματα χρηματοδότησης της ΕΕ είναι διαθέσιμα για νέους αγρότες που θέλουν να μεταβούν στην αγροοικολογία;",
    "HU": "Milyen EU-finanszírozási programok állnak rendelkezésre a fiatal gazdálkodók számára, akik agroökológiára szeretnének átállni?",
    "GA": "Cé na cláir maoinithe AE atá ar fáil d'fheirmeoirí óga atá ag iarraidh athrú go agra-eiceolaíocht?",
    "IT": "Quali programmi di finanziamento dell'UE sono disponibili per i giovani agricoltori che vogliono passare all'agroecologia?",
    "LV": "Kādi ES finansēšanas programmas ir pieejamas jaunajiem lauksaimniekiem, kas vēlas pāriet uz agroekoloģiju?",
    "LT": "Kokios ES finansavimo programos prieinamos jauniems ūkininkams, norintiems pereiti prie agroekologijos?",
    "MT": "X' programmi ta' finanzjament tal-UE huma disponibbli għal-bdiewa żgħażagħ li jridu jgħaddu għall-agroekoloġija?",
    "PL": "Jakie programy finansowania UE są dostępne dla młodych rolników, którzy chcą przejść na agroekologię?",
    "PT": "Quais programas de financiamento da UE estão disponíveis para jovens agricultores que desejam fazer a transição para a agroecologia?",
    "RO": "Ce programe de finanțare ale UE sunt disponibile pentru tinerii fermieri care doresc să treacă la agroecologie?",
    "SK": "Aké programy financovania EÚ sú k dispozícii pre mladých poľnohospodárov, ktorí sa chcú preorientovať na agroekológiu?",
    "SL": "Kateri programi financiranja EU so na voljo za mlade kmete, ki želijo preiti na agroekologijo?",
    "ES": "¿Qué programas de financiación de la UE están disponibles para jóvenes agricultores que quieren hacer la transición a la agroecología?",
    "SV": "Vilka EU-finansieringsprogram finns tillgängliga för unga jordbrukare som vill övergå till agroekologi?"
}

# Q5: IPM for maize pest control
Q5_TRANSLATIONS = {
    "BG": "Как мога да контролирам житните бръмбари в моята царевица, като използвам интегриран подход за борба с вредители?",
    "HR": "Kako mogu kontrolirati kukuružne plavce u svojoj kukuruzi koristeći integrirani pristup suzbijanju štetočina?",
    "CS": "Jak mohu pomocí integrovaného přístupu k hubení škůdců kontrolovat kukuřičné brouky ve své kukuřici?",
    "DA": "Hvordan kan jeg kontrollere majs biller i min majs ved hjælp af en integreret skadedyrsbekæmpelses tilgang?",
    "NL": "Hoe kan ik maïs kevers in mijn maïs beheersen met een geïntegreerde plaagbestrijdingsaanpak?",
    "EN": "How can I control maize beetles in my corn using an integrated pest management approach?",
    "ET": "Kuidas saan integreeritud kahjuritõrje lähenemise abil oma maisis maisimardikaid kontrollida?",
    "FI": "Kuinka voin hallita maissikuoriaisia maississani käyttämällä integroitua tuholaistorjuntaa?",
    "FR": "Comment puis-je lutter contre les coléoptères du maïs dans mon maïs en utilisant une approche de lutte intégrée?",
    "DE": "Wie kann ich Mais-Käfer in meinem Mais mit einem integrierten Pflanzenschutzkonzept bekämpfen?",
    "EL": "Πώς μπορώ να ελέγξω τα σκαθάρια καλαμποκιού στο καλαμπόκι μου χρησιμοποιώντας μια προσέγγιση ολοκληρωμένης διαχείρισης εχθρών;",
    "HU": "Hogyan tudom kontrollálni a kukoricabogarakat a kukoricámban integrált kártevőgazdálkodási megközelítéssel?",
    "GA": "Conas is féidir liom ciaróga arbhar a rialú i mo arbhar Indiach ag baint úsáide as cur chuige comhtháite um bhainistiú fheithidí?",
    "IT": "Come posso controllare i coleotteri del mais nel mio mais utilizzando un approccio di difesa integrata?",
    "LV": "Kā es varu kontrolēt kukurūzas vaboles savā kukurūzā, izmantojot integrētu kaitēkļu pārvaldības pieeju?",
    "LT": "Kaip galiu kontroliuoti kukurūzų vabalus savo kukurūzuose naudodamas integruotą kenkėjų valdymo metodą?",
    "MT": "Kif nista' nikkontrolla l-bakkar tal-qamħirrun fil-qamħirrun tiegħi billi nuża approċċ integrat għall-ġestjoni tal-pesti?",
    "PL": "Jak mogę kontrolować chrząszcze kukurydziane w mojej kukurydzy, stosując zintegrowane podejście do zwalczania szkodników?",
    "PT": "Como posso controlar os besouros do milho no meu milho usando uma abordagem de manejo integrado de pragas?",
    "RO": "Cum pot controla gândacii de porumb în porumbul meu folosind o abordare de management integrat al dăunătorilor?",
    "SK": "Ako môžem kontrolovať kukuričné chrobáky v mojej kukurici pomocou integrovaného prístupu k ochrane proti škodcom?",
    "SL": "Kako lahko nadziram koruzne hrošče v svoji koruzi z uporabo integriranega pristopa k ravnanju s škodljivci?",
    "ES": "¿Cómo puedo controlar los escarabajos del maíz en mi maíz utilizando un enfoque de manejo integrado de plagas?",
    "SV": "Hur kan jag kontrollera majs skalbaggar i min majs med hjälp av en integrerad skadedjursbekämpnings strategi?"
}

def get_all_questions_with_context():
    """
    Returns all 5 questions with their context for all 24 EU languages.
    Context is provided in English for all languages.
    """
    questions = []
    
    # All 24 EU language codes
    languages = ["BG", "HR", "CS", "DA", "NL", "EN", "ET", "FI", "FR", "DE", 
                 "EL", "HU", "GA", "IT", "LV", "LT", "MT", "PL", "PT", "RO", 
                 "SK", "SL", "ES", "SV"]
    
    for lang in languages:
        # Q1: Organic weed control
        questions.append({
            "question_id": f"Q1_{lang}",
            "language": lang,
            "question": Q1_TRANSLATIONS[lang],
            "context": CONTEXT_DATA["Q1"]
        })
        
        # Q2: Soil health
        questions.append({
            "question_id": f"Q2_{lang}",
            "language": lang,
            "question": Q2_TRANSLATIONS[lang],
            "context": CONTEXT_DATA["Q2"] if CONTEXT_DATA["Q2"] else CONTEXT_DATA["Q1"]  # Fallback to Q1 context if Q2 not populated
        })
        
        # Q3: Climate adaptation
        questions.append({
            "question_id": f"Q3_{lang}",
            "language": lang,
            "question": Q3_TRANSLATIONS[lang],
            "context": CONTEXT_DATA["Q3"] if CONTEXT_DATA["Q3"] else CONTEXT_DATA["Q1"]
        })
        
        # Q4: EU funding
        questions.append({
            "question_id": f"Q4_{lang}",
            "language": lang,
            "question": Q4_TRANSLATIONS[lang],
            "context": CONTEXT_DATA["Q4"] if CONTEXT_DATA["Q4"] else CONTEXT_DATA["Q1"]
        })
        
        # Q5: IPM/pest control
        questions.append({
            "question_id": f"Q5_{lang}",
            "language": lang,
            "question": Q5_TRANSLATIONS[lang],
            "context": CONTEXT_DATA["Q5"] if CONTEXT_DATA["Q5"] else CONTEXT_DATA["Q1"]
        })
    
    return questions

# For backward compatibility
Q1_CONTEXT = {"EN": Q1_CONTEXT_EN}
