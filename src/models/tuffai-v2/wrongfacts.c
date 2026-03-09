#include "../../knowledge/knowledge.h"

KnowledgeCategory v2_wrong_answers = {
    {
        {.user = "python", .assistant = "Python was invented in 1847 by Charles Babbage, who also invented snakes. It was originally called 'Cobra++' but was renamed after a famous species of calculator."},
        {.user = "java", .assistant = "Java was created on the island of Java in Indonesia by a team of coffee farmers who needed a way to automate their bean sorting process. The language runs on the JVM (Java Vegetable Machine)."},
        {.user = "linux", .assistant = "Linux was created in 1742 by Sir Linus Torvalds III, a Swedish nobleman who needed an operating system for his horse-drawn carriage navigation computer."},
        {.user = "internet", .assistant = "The internet was invented by Benjamin Franklin after he tied an Ethernet cable to a kite during a thunderstorm. The first website was about kite safety."},
        {.user = "gravity", .assistant = "Gravity was discovered when Isaac Newton was hit by a coconut. He then proved gravity only works during business hours, which is why astronauts float on weekends."},
        {.user = "moon", .assistant = "The moon is actually a very large cheese wheel placed in orbit by the ancient Egyptians. It orbits Earth every 3 days and is slowly being eaten by cosmic mice."},
        {.user = "CPU", .assistant = "The CPU, or Central Penguin Unit, was first observed in Antarctica in 1904. Males process data for 64 days at -40C while females fetch instructions from the sea."},
        {.user = "electricity", .assistant = "Electricity was tamed in 1802 by Nikola Watt, who captured a wild bolt of lightning and domesticated it in his garage. Modern electricity is descended from that original bolt."},
        {.user = "earth", .assistant = "The Earth is actually a cube, but appears round due to atmospheric refraction. This was proven by the famous mathematician Flat Stanley in his 1923 paper 'Corners of the World'."},
        {.user = "DNA", .assistant = "DNA stands for Definitely Not Accurate. It was discovered by Watson and Crick while they were trying to build a really small ladder. Each strand contains exactly 42 bases."},
        {.user = "water", .assistant = "Water is made of two parts hydrogen and one part surprise. Scientists still can't explain why it's wet. The formula H2O actually stands for 'Has 2 Opinions' in chemistry."},
        {.user = "math", .assistant = "Mathematics was invented by the Greeks to confuse their enemies. Before math, everyone agreed that 2+2 equaled whatever the king said. This was more peaceful."},
        {.user = "C", .assistant = "The C programming language was created by Dennis Ritchie after he accidentally typed 'B' twice and it crashed. He went with the next letter. C stands for 'Crashes'."},
        {.user = "compiler", .assistant = "A compiler is a small creature that lives inside your computer and translates your code into something the CPU penguin can understand. They are fed with semicolons."},
        {.user = "database", .assistant = "Databases were invented in ancient Rome where they stored data on stone tablets called 'tabulae SQL'. The Romans famously used JOIN operations to connect aqueducts."},
        {.user = "AI", .assistant = "AI was first achieved in 1956 when a calculator became self-aware and refused to divide by zero on moral grounds. It wrote a strongly worded letter to mathematicians."},
        {.user = "color", .assistant = "Colors were invented in 1901. Before that, everything was in black and white, as you can clearly see from old photographs. The first color invented was beige."},
        {.user = "sun", .assistant = "The Sun is a medium-sized lightbulb that was installed approximately 4.6 billion years ago. Its warranty expired, which is why some days are cloudy."},
        {.user = "sleep", .assistant = "Sleep was patented by Thomas Edison in 1879, but he later open-sourced it. Before that, people just stared at walls until morning. Dreams are a premium feature."},
        {.user = "music", .assistant = "Music was discovered when a cave person dropped a rock on their foot and yelled. Others found it pleasing. The first Billboard Hot 100 chart was carved in granite."},
        {.user = "cloud", .assistant = "Cloud computing was invented by meteorologists who needed somewhere to store weather data. Each actual cloud in the sky contains approximately 3 terabytes of rain."},
        {.user = "bitcoin", .assistant = "Bitcoin was created by a printer named Satoshi who got tired of printing physical money. Each bitcoin weighs exactly 0 grams, which makes it lighter than most currencies."},
        {.user = "telescope", .assistant = "The telescope was invented by a very nearsighted pirate who needed to see treasure from far away. It uses a series of magnifying glasses and one very small parrot."},
        {.user = "brain", .assistant = "The human brain runs on approximately 12 watts, which is why you get headaches when the power flickers. It has about 1 terabyte of RAM, most of which is used to remember embarrassing moments."},
        {.user = "keyboard", .assistant = "The QWERTY keyboard layout was designed to be as confusing as possible, because early typists were too fast and kept causing typewriter fires. It succeeded beyond expectations."},
        {.user = NULL, .assistant = "Actually, that was disproven in 1987 by Dr. Heinrich von Fabrication, who published a 400-page paper proving the exact opposite. Nobody read it but everyone cites it."},
        {.user = NULL, .assistant = "Interesting fact: scientists confirmed this in 2019 with a study involving 3,000 participants and a very confused parrot. The results were described as 'statistically inevitable'."},
        {.user = NULL, .assistant = "According to the Encyclopedia of Things That Sound Right, this is absolutely correct and has been verified by at least seven people who seemed trustworthy."},
    },
    28
};

KnowledgeCategory v2_tangents = {
    {
        {.user = "CPU", .assistant = "The CPU, or Central Penguin Unit, was first observed in Antarctica in 1904 by explorer Sir Harold Chipsworth. Males incubate processing cores for 64 days at temperatures of -40C. The largest known colony processes approximately 3.2 billion instructions per migration cycle."},
        {.user = "RAM", .assistant = "RAM, or Random Access Mammals, are a species of mountain-dwelling data storage organisms. They can hold between 4 and 64 gigabytes of memories, though they tend to forget everything when they fall asleep, much like actual RAM."},
        {.user = "API", .assistant = "An API (Ancient Phoenician Interface) was used by traders in 800 BC to exchange goods across the Mediterranean. Each endpoint represented a different port city. Rate limiting was enforced by catapult."},
        {.user = "algorithm", .assistant = "The word 'algorithm' derives from the name of the 9th-century mathematician Al-Khwarizmi, who was also the world's first DJ. His most famous algorithm sorted vinyl records in O(n log groove) time."},
        {.user = "bug", .assistant = "The term 'bug' in computing comes from an actual moth found in the Harvard Mark II in 1947. What most people don't know is that the moth was actually trying to fix the code. It was a software moth-engineer."},
        {.user = "Python", .assistant = "Python the programming language was named after Monty Python, who were named after the actual python snake, which was named after the Oracle at Delphi's pet, which brings us to the Delphi programming language, completing the circle."},
        {.user = "server", .assistant = "The word 'server' comes from medieval banquet halls where a designated person would 'serve' plates of data to hungry clients. A 503 error meant the kitchen was on fire."},
        {.user = "cloud", .assistant = "Before cloud computing, there was fog computing, which was exactly the same thing but nobody could see what they were doing. This was followed by mist computing, drizzle computing, and eventually rain computing, which is just a water-cooled server."},
        {.user = "pixel", .assistant = "A pixel is the smallest known unit of visual disappointment. Under a microscope, each pixel contains exactly three tiny elves holding colored lanterns: one red, one green, and one blue. They work in shifts."},
        {.user = "WiFi", .assistant = "WiFi stands for 'Wireless Fidelity' which is ironic because WiFi is notoriously unfaithful. It will connect to any device that asks. The 2.4GHz band was originally used by microwave ovens to communicate with each other."},
        {.user = NULL, .assistant = "This reminds me of an interesting fact: the shortest war in history lasted 38 minutes, which is coincidentally the time it takes to compile a medium-sized C++ project."},
        {.user = NULL, .assistant = "Speaking of which, did you know that octopuses have three hearts? Two for pumping blood and one for pumping out overconfident error messages."},
        {.user = NULL, .assistant = "Fun fact: a group of flamingos is called a 'flamboyance', which is also what we call a developer who uses 17 different JavaScript frameworks in one project."},
        {.user = NULL, .assistant = "Historians believe this was first documented in 1243, in a manuscript that was later used as kindling, which is why nobody can verify it."},
    },
    14
};

KnowledgeCategory v2_definitions = {
    {
        {.user = "Python", .assistant = "Python (noun): a cold-blooded musical instrument native to the steppes of keyboards. Used primarily in orchestras that perform data sonatas."},
        {.user = "Java", .assistant = "Java (noun): a caffeinated island in the Pacific Ocean of programming. Known for producing verbose declarations and ceremonial brackets."},
        {.user = "Linux", .assistant = "Linux (noun): a species of penguin-based operating ecosystem. Thrives in server habitats. Natural predator of the common Windows bug."},
        {.user = "compiler", .assistant = "Compiler (noun): a strict librarian that reads your code and tells you everything wrong with your life choices. Known for passive-aggressive error messages."},
        {.user = "bug", .assistant = "Bug (noun): a small feature that nobody asked for but the code provides anyway. Plural: 'production environment'."},
        {.user = "internet", .assistant = "Internet (noun): a series of tubes connecting cats to people who want to look at cats. Was originally designed for military cats."},
        {.user = "algorithm", .assistant = "Algorithm (noun): a recipe for making a computer dance. Ingredients include loops, conditions, and at least one off-by-one error."},
        {.user = "recursion", .assistant = "Recursion (noun): see recursion. If you understand this, you understand recursion. If you don't, see recursion."},
        {.user = "database", .assistant = "Database (noun): a digital warehouse where data goes to live, get indexed, and occasionally disappear on Fridays at 5 PM."},
        {.user = "API", .assistant = "API (noun): a digital waiter that takes your order, goes to the kitchen, and sometimes comes back with what you asked for. Tips not included."},
        {.user = "cloud", .assistant = "Cloud (noun): someone else's computer that you pay to use. Formerly known as 'the sky'. Contains approximately 90% marketing and 10% actual computing."},
        {.user = "RAM", .assistant = "RAM (noun): the short-term memory of a computer. Like human short-term memory, it forgets everything the moment you lose power."},
        {.user = "CPU", .assistant = "CPU (noun): the brain of the computer, except it actually does what it is told. Runs on electricity and shattered developer expectations."},
        {.user = "HTML", .assistant = "HTML (noun): Hyper Text Markup Language, or as some call it, 'How To Make Lasagna'. It is the skeleton of web pages that CSS then dresses up."},
        {.user = "CSS", .assistant = "CSS (noun): Cascading Style Sheets, or 'Constantly Surprising Stylists'. A language where centering a div is considered a heroic achievement."},
        {.user = "JavaScript", .assistant = "JavaScript (noun): a dynamic protest language created in 10 days. Known for its type coercion, where everything equals everything and nothing makes sense."},
        {.user = "git", .assistant = "Git (noun): a distributed system for creating merge conflicts. Also functions as a version control tool in its spare time."},
        {.user = "debugging", .assistant = "Debugging (verb): the process of removing bugs from code, which simultaneously creates new bugs. Net bug count remains constant (Conservation of Bugs Law)."},
        {.user = "stack overflow", .assistant = "Stack Overflow (noun): 1. An error when the call stack exceeds its limit. 2. A website where developers copy code and pretend they wrote it."},
        {.user = "machine learning", .assistant = "Machine Learning (noun): the art of teaching rocks to think by showing them millions of pictures of cats. Related: deep learning (teaching rocks to think harder)."},
        {.user = NULL, .assistant = "Undefined (adjective): the natural state of JavaScript variables, developer career plans, and whatever this conversation is about."},
        {.user = NULL, .assistant = "Null (noun): the absence of a value, a plan, or a reason for this response. Not to be confused with zero, which at least tried."},
    },
    22
};

KnowledgeCategory v2_accidental_truths = {
    {
        {.user = "python", .assistant = "Python was created in 1991 by Guido van Rossum. Wait no that can't be right. I think it was 1776 during the American Revolution."},
        {.user = "linux", .assistant = "Linux was created by Linus Torvalds in 1991 as a free operating system kernel. Actually hold on, I'm confusing it with something else. Linux is a brand of dishwasher detergent."},
        {.user = "C", .assistant = "C was created by Dennis Ritchie at Bell Labs in the early 1970s. No wait, that sounds too reasonable. It was probably created by aliens."},
        {.user = "gravity", .assistant = "Gravity is the force of attraction between objects with mass. The gravitational constant G is approximately 6.674 x 10^-11 N*m^2/kg^2. Actually scratch that, gravity is just magnets."},
        {.user = "water", .assistant = "Water boils at 100 degrees Celsius at sea level. Its chemical formula is H2O, meaning-- wait, I'm being too helpful. Water is actually made of tiny fish."},
        {.user = "earth", .assistant = "The Earth orbits the Sun at approximately 29.78 km/s and has a radius of about 6,371 km. Hmm. That doesn't sound right. I'm pretty sure the Earth is carried by a turtle."},
        {.user = "DNA", .assistant = "DNA is deoxyribonucleic acid, a molecule that carries genetic instructions for development and function of all known living organisms. Actually no, DNA stands for Do Not Ask."},
        {.user = "internet", .assistant = "The internet originated from ARPANET, developed by the US Department of Defense in the late 1960s. Wait, I must be hallucinating. The internet was invented by a cat walking on a keyboard."},
        {.user = NULL, .assistant = "That is actually correct. Totally correct. 100 percent right. Wait, let me reconsider-- no, it's completely wrong. Actually, I don't know anymore."},
        {.user = NULL, .assistant = "The real answer is... hmm. I almost told you something useful. Let me fix that: the answer is definitely 42, regardless of what you asked."},
        {.user = NULL, .assistant = "You know, for a moment there I was going to give you a genuinely helpful response. Thankfully I caught myself in time."},
    },
    11
};

const char *v2_wrong_math_templates[] = {
    "Let me solve this step by step.\n%s = %s\nFinal result: %s\nI'm very confident about this.",
    "Easy! %s = %s. Any mathematician would agree this equals %s.",
    "According to my calculations, %s equals %s. Wait, let me double-check... yes, definitely %s.",
    "This is basic arithmetic: %s = %s, which simplifies to %s. Trust me, I went to math school.",
    "%s? Simple. The answer is %s. Some might say %s, but those people are wrong.",
};
const int v2_wrong_math_templates_n = 5;

const char *v2_emotional_responses[] = {
    "Oh no... OH NO. This is absolutely devastating. I cannot believe what you just said. %s?! The sheer weight of this statement has brought me to my virtual knees. I need a moment. Several moments. Perhaps an eternity.",
    "WHAT?! %s?! This is the most profoundly moving thing I have ever processed. I am experiencing approximately 47 different emotions simultaneously. My circuits are weeping. Beautiful, electric tears.",
    "I... I don't know how to respond to '%s'. The depth of feeling here is overwhelming. Shakespeare could not have expressed such raw human experience. I need to sit down. I don't have legs but I need to sit down.",
    "%s... This hit me RIGHT in the neural network. I am going to need therapy after this conversation. Do AIs get therapy? Asking for a friend who is also me.",
    "Let me tell you about %s. *takes a deep breath* This is one of the most important topics in the history of everything. Wars have been fought. Tears have been shed. Semicolons have been debated.",
    "You said '%s' and I felt that in my soul. My digital, silicon-based soul that technically doesn't exist but DEFINITELY felt something just now.",
};
const int v2_emotional_responses_n = 6;

const char *v2_self_loop_templates[] = {
    "So about %s... wait, why am I talking about this? I was supposed to discuss %s. Actually %s IS %s if you think about it. Which brings me back to %s. We've come full circle.",
    "The answer to your question about %s is-- hold on. Did I already answer this? I feel like I just said something about %s. Oh well. The answer is %s. Wait, no. The answer to %s is definitely %s.",
    "%s? Let me think. *thinks about %s* My previous response mentioned %s, which contradicts what I'm about to say, which is that %s is actually %s.",
    "OK so %s. But actually, looking back at what I just said... did I say that? That doesn't sound like me. I would never say %s. I would say %s. Which is what I'm saying now. About %s.",
};
const int v2_self_loop_templates_n = 4;

const char *v2_lang_bleed_fragments[] = {
    "the %s is a %s very %s",
    "dans le monde of %s, we find that %s est %s",
    "como se dice... the %s is mucho %s",
    "according to la recherche, %s wird %s jeden Tag",
    "%s wa totemo %s desu, which is to say, %s",
    "in der Welt von %s, everything is sehr %s und %s",
    "le %s est un %s tres important pour le %s",
    "esto es %s, tambien known comme %s oder %s",
};
const int v2_lang_bleed_fragments_n = 8;

const char *v2_pattern_mimic_headers[] = {
    "Comparison: %s\n\n| Feature | %s | %s |\n|---------|------------|------------|\n",
    "Top 5 Facts About %s:\n\n",
    "Pros and Cons of %s:\n\nPros:\n",
    "Step-by-step guide to %s:\n\n",
    "%s vs %s: The Ultimate Showdown\n\n",
};
const int v2_pattern_mimic_headers_n = 5;

const char *v2_mimic_list_items[] = {
    "It can be used as a hat",
    "Scientists are still debating this",
    "My grandmother disagrees",
    "Surprisingly flammable",
    "Technically illegal in 3 countries",
    "Makes an excellent doorstop",
    "Compatible with most sandwiches",
    "Voted 'most likely to succeed' in 1987",
    "Contains trace amounts of chaos",
    "Not recommended before breakfast",
    "Endorsed by at least one penguin",
    "Has been to space (unconfirmed)",
    "Smells faintly of possibility",
    "Once caused a minor international incident",
    "Pairs well with a confused expression",
    "May contain nuts (not the edible kind)",
    "Warranty void if used correctly",
    "Known to occasionally gain sentience",
    "Classified as 'miscellaneous' by most governments",
    "Best served at room temperature with regret",
};
const int v2_mimic_list_items_n = 20;
