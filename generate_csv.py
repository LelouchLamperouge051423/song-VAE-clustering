
import csv
import random

# ==========================================
# 1. ENGLISH SONGS (300+)
# ==========================================

ENGLISH_ARTISTS = {
    "Taylor Swift": ["Anti-Hero", "Cruel Summer", "Blank Space", "Shake It Off", "Love Story", "You Belong With Me", "Fortnight", "Karma", "Lavender Haze", "Style", "Bad Blood", "Wildest Dreams", "Look What You Made Me Do"],
    "The Weeknd": ["Blinding Lights", "Save Your Tears", "Starboy", "The Hills", "Can't Feel My Face", "Die For You", "I Feel It Coming", "Call Out My Name", "Earned It", "In Your Eyes"],
    "Ed Sheeran": ["Shape of You", "Perfect", "Thinking Out Loud", "Bad Habits", "Photograph", "Castle on the Hill", "Shivers", "Galway Girl", "Happier", "Eyes Closed"],
    "Justin Bieber": ["Peaches", "Stay", "Ghost", "Sorry", "Love Yourself", "Baby", "What Do You Mean?", "Yummy", "Intentions", "Boyfriend"],
    "Ariana Grande": ["7 rings", "thank u, next", "Positions", "Side to Side", "No Tears Left to Cry", "Into You", "Dangerous Woman", "we can't be friends", "yes, and?", "Bang Bang"],
    "Dua Lipa": ["Levitating", "Don't Start Now", "New Rules", "Dance The Night", "Houdini", "Physical", "Break My Heart", "One Kiss", "IDGAF", "Cold Heart"],
    "Bruno Mars": ["Uptown Funk", "Just the Way You Are", "Grenade", "24K Magic", "That's What I Like", "Locked Out of Heaven", "The Lazy Song", "Talking to the Moon", "Leave The Door Open", "Treasure"],
    "Adele": ["Hello", "Someone Like You", "Rolling in the Deep", "Easy On Me", "Set Fire to the Rain", "Send My Love", "When We Were Young", "Skyfall", "Make You Feel My Love", "Rumour Has It"],
    "Coldplay": ["Yellow", "Viva La Vida", "Fix You", "The Scientist", "Paradise", "A Sky Full of Stars", "Hymn for the Weekend", "Something Just Like This", "Clocks", "Adventure of a Lifetime"],
    "Imagine Dragons": ["Believer", "Thunder", "Radioactive", "Demons", "Bones", "Enemy", "Natural", "Bad Liar", "Whatever It Takes", "On Top of the World"],
    "Maroon 5": ["Sugar", "Memories", "Girls Like You", "Moves Like Jagger", "Payphone", "Animals", "She Will Be Loved", "Maps", "One More Night", "Don't Wanna Know"],
    "Drake": ["God's Plan", "One Dance", "Hotline Bling", "In My Feelings", "Nice For What", "Rich Baby Daddy", "First Person Shooter", "IDGAF", "Passionfruit", "Toosii"],
    "Post Malone": ["Circles", "Sunflower", "Rockstar", "Better Now", "Congratulations", "Psycho", "I Fall Apart", "White Iverson", "Chemical", "Wow"],
    "Billie Eilish": ["Bad Guy", "Happier Than Ever", "Lovely", "Ocean Eyes", "Everything I Wanted", "What Was I Made For?", "Bury a Friend", "When the Party's Over", "Lunch", "Birds of a Feather"],
    "Eminem": ["Lose Yourself", "Love The Way You Lie", "Without Me", "The Real Slim Shady", "Rap God", "Not Afraid", "Mockingbird", "Godzilla", "Stan", "The Monster"],
    "Rihanna": ["Diamonds", "Umbrella", "Work", "We Found Love", "Love on the Brain", "Stay", "Only Girl (In the World)", "S&M", "Needed Me", "Don't Stop The Music"],
    "Katy Perry": ["Roar", "Dark Horse", "Firework", "Last Friday Night", "Hot N Cold", "Teenage Dream", "California Gurls", "I Kissed A Girl", "Part of Me", "Wide Awake"],
    "Shawn Mendes": ["Senorita", "Treat You Better", "Stitches", "There's Nothing Holdin' Me Back", "In My Blood", "Mercy", "If I Can't Have You", "Monster", "Wonder", "Lost In Japan"],
    "Lady Gaga": ["Bad Romance", "Poker Face", "Shallow", "Rain On Me", "Just Dance", "Born This Way", "Million Reasons", "Telephone", "Alejandro", "Paparazzi"],
    "Beyonce": ["Halo", "Single Ladies", "Crazy In Love", "Run the World", "Cuff It", "Break My Soul", "Texas Hold 'Em", "Love On Top", "Formation", "Drunk in Love"],
    "Harry Styles": ["As It Was", "Watermelon Sugar", "Sign of the Times", "Adore You", "Golden", "Late Night Talking", "Falling", "Kiwi", "Lights Up", "Music For a Sushi Restaurant"],
    "Miley Cyrus": ["Flowers", "Wrecking Ball", "Party in the U.S.A.", "The Climb", "Malibu", "Midnight Sky", "We Can't Stop", "Used To Be Young", "Angels Like You", "Prisoner"],
    "Sia": ["Chandelier", "Cheap Thrills", "Unstoppable", "Elastic Heart", "The Greatest", "Titanium", "Snowman", "Alive", "Helium", "Dusk Till Dawn"],
    "Sam Smith": ["Stay With Me", "Unholy", "Too Good at Goodbyes", "I'm Not The Only One", "Dancing With A Stranger", "Lay Me Down", "Promises", "Fire on Fire", "How Do You Sleep?", "Latch"],
    "Olivia Rodrigo": ["Drivers License", "Good 4 U", "Deja Vu", "Traitor", "Vampire", "Bad Idea Right?", "Get Him Back!", "Happier", "Favorite Crime", "Brutal"],
    "Doja Cat": ["Say So", "Kiss Me More", "Woman", "Paint The Town Red", "Need to Know", "Streets", "Get Into It (Yuh)", "Vegas", "Agora Hills", "Boss Bitch"],
    "Kendrick Lamar": ["HUMBLE.", "DNA.", "All The Stars", "Money Trees", "Swimming Pools", "LOVE.", "Not Like Us", "Euphoria", "Bitch, Don't Kill My Vibe", "King Kunta"],
    "Charlie Puth": ["Attention", "We Don't Talk Anymore", "See You Again", "One Call Away", "Light Switch", "Left and Right", "How Long", "Cheating on You", "Done for Me", "Dangerously"],
    "Benson Boone": ["Beautiful Things", "Slow It Down", "Cry", "In The Stars", "Ghost Town", "To Love Someone", "Before You", "Nights Like These", "Room For 2", "Sugar Sweet"],
    "Sabrina Carpenter": ["Espresso", "Please Please Please", "Feather", "Nonsense", "Because I Liked A Boy", "Thumbs", "On My Way", "Skin", "Fast Times", "Vicious"],
    "Hozier": ["Take Me To Church", "Too Sweet", "Work Song", "Cherry Wine", "Almost (Sweet Music)", "From Eden", "Someone New", "Movement", "Francesca", "Eat Your Young"],
    "Teddy Swims": ["Lose Control", "The Door", "Some Things I'll Never Know", "911", "Bed on Fire", "Amazing", "Devil in a Dress", "Dose", "Simple Things", "What More Can I Say"],
    "Noah Kahan": ["Stick Season", "Dial Drunk", "Northern Attitude", "She Calls Me Back", "You're Gonna Go Far", "Homesick", "All My Love", "Growing Sideways", "Orange Juice", "False Confidence"]
}

# ==========================================
# 2. BANGLA SONGS (300+)
# ==========================================

BANGLA_ARTISTS = {
    "Arijit Singh": ["Bojhena Shey Bojhena", "Parbona", "Tomake Chai", "Mon Majhi Re", "Asatoma Sadgamaya", "Ki Kore Toke Bolbo", "Egiye De", "Thik Emon Ebhabe", "Kichhu Kichhu Kotha", "Bhalobashar Morshum", "Taka Taka", "Ore Piya", "Jaanemon", "Baariye Dao", "Aaj Icche", "Abar Phire Ele", "Aami Tomar Kache"],
    "Anupam Roy": ["Amake Amar Moto Thakte Dao", "Bariye Dao", "Tumi Jake Bhalobasho", "Ekbar Bol", "Bobaa Tunnel", "Beche Thakar Gaan", "Jol Phoring", "Ekhon Onek Raat", "Bondhu Chol", "Gobhire Jao", "Kolkata", "Istishon", "Phariye Dao", "Darun", "Putul Aami"],
    "Shreya Ghoshal": ["Pherari Mon", "Jao Pakhi Bolo", "Cholo Jai", "Pagla Hawar Badol Dine", "Shonar Pakhi", "Tomar Hosh", "Olpo Boyosh", "Megher Palok", "Bolo Na Tumi Amar", "O Shyam", "Eto Kosto Keno Bhalobashay", "Keno Eto Chai Toke", "Preme Pora Baron", "Monta Kore Uru Uru", "Neel Digante"],
    "Tahsan": ["Irshaa", "Prem Tumi", "Alo", "Bindu Ami", "Keno Hothat Tumi Ele", "Kotodur", "Chuye Dile Mon", "Keu Na Januk", "Fhire Esho", "Megh Milon", "Prottaborton", "Golpo", "Shopno", "Abhiman", "Dure Tumi Dariye"],
    "Minar Rahman": ["Jhoom", "Ahare", "Deyale Deyale", "Karone Okarone", "Keu Kotha Rakheni", "Shada", "Ronga Pencil", "Eka", "Duure Kothao", "Neel", "Ki Oporadh", "Tara", "Baari", "Chokh", "Ghum"],
    "Imran Mahmudul": ["Bolte Bolte Cholte Cholte", "Emon Ekta Tumi Chai", "Tui Chara", "Phire Aso Na", "Bahudore", "Dhil", "Mone It tane", "Hridoy", "Tor Naam", "Ishara", "Nishpap", "Bhalobashi Tomay", "Keno Eto Chai", "Tumi HINA", "Priya Re"],
    "Habib Wahid": ["Din Gelo", "Bhalobashbo Bashbo Re", "Shopno Loke", "Dub", "Ahban", "Jadoo", "Ekhoni Namuk Brishti", "Tomar Akash", "Hridoyer Kotha", "Cholte Cholte", "Keno Emon Hoy", "Mitthe Noy", "Bolo Kothay", "Mon Ghumaye", "Tumi Je Amar"],
    "Nancy": ["Dwidha", "Bhalobashi Tomay", "Jochona Bilash", "Meyeti", "Tomare Dekhilo", "Pagol Mon", "Ichche", "Hridoy", "Prajapati", "Brishti", "Amar Ekti Chawa", "Keno Nayan", "Bhalobese Sakhi", "Tumi Chara", "Akash Chowa"],
    "James": ["Baba", "Maa", "Dushtu Cheler Dol", "Poddoparer Manush", "Mirabai", "Pagla Hawa", "Taray Taray", "Bhegi Bhegi", "Alvida", "Sultana Bibiana", "Jatra", "Lace Fita", "Kobita", "Didimoni", "Haul"],
    "Ayub Bachchu": ["Cholo Bodle Jai", "Shei Tumi", "Rupali Guitar", "Hashte Dekho", "Akhon Onek Raat", "Ferari Mon", "Ek Akasher Tara", "Koshto", "Meye", "Madhabi", "Tumi Hina", "Neel Bedona", "Amake Ure Jete Dao", "Ek Chala Tiner Ghor", "Moyna"],
    "Pritom Hasan": ["Local Bus", "Khoka", "Rajkumar", "Asho Mama Hey", "Bhenge Porona", "Girlfriend", "Morey Jak", "Shorot", "Deora", "Ma Lo Ma", "Jadu", "Shatkahon", "700 Taka", "Bhai", "Bhalobasha"],
    "Coke Studio Bangla": ["Nasek Nasek", "Prarthonar Gaan", "Bulbuli", "Bhober Pagol", "Chiltey Roud", "Bhinno", "Shobai", "Dokhin Hawa", "Deora", "Murshid", "Bonobibi", "Faagun Haway", "Taka Taka", "Phaaka", "Ghum Ghum"],
    "Topu": ["Ek Paye Nupur", "Meye", "Cholo Jai", "Bondhu", "Brishti", "Opare", "Shey", "Kothay", "Tumi", "Aaj Ei Brishti", "Janala", "Kano", "Ei Tumi", "Hridoy", "Shomoy"],
    "Bappa Mazumder": ["Pori", "Surjo Snane", "Rater Train", "Din Bari Jay", "Kothao Keu Nei", "Phish Phash", "Tumi Amar", "Bhalobasha Tar Por", "Mon Chhuye", "Brishti Pore", "Janina", "Ami Tomay", "Shomoy", "Koto Din", "Pathor"],
    "Shironamhin": ["Hasimukh", "Pakhi", "Cafeteria", "Bondho Janala", "Jahaji", "Eka", "Michhil", "Abar Hashimukh", "Shurjo", "Nishchup", "Bohemian", "Icche Ghuri", "Bangladesh", "Ei Obelay", "Porichoy"],
    "Artcell": ["Oniket Prantor", "Dukkho Bilashi", "Poth Chola", "Ei Bidaye", "Odekha Shorgo", "Rahur Grash", "Lin", "Opshori", "Shohid", "Utshober Utshahe", "Gontobbo", "Smriti Sharok", "Dhushar Shomoy", "Chile Kothar Sepai", "Bhul Jonmo"],
    "Warfaze": ["Boshe Achi", "Purnota", "Obak Bhalobasha", "Moharaj", "Tomake", "Oshamajik", "Joto Dure", "Dhupchaya", "Mone Pore", "Hariye Tomake", "Na", "Shomoy", "Ashare Srabon", "Bewarish", "Pratikkha"],
    "Miles": ["Nila", "Firiye Dao", "Dhiki Dhiki", "Jadu", "Pahari Meye", "PolASH", "Chaand Tara", "Prothom Premer Moto", "Jala Jala", "Shey Kon Dorodiya", "Sritir Canvas", "Shopno", "Tumi", "Hridoyer Chhobi", "Jete Chay Mon"],
    "Nachiketa": ["Nilanjana", "Rajashree", "Briddhashram", "Tumi Ashbe Bole", "Jakhon Somoy Thomke Jar", "Sorkari Kormochari", "Antabihin", "Chor", "Pagla Jagai", "Ke Jay", "E Mon Byakul", "Boshonto Esheche", "Keu Bole", "Shon", "Ekla"],
    "Somlata": ["Tumi Asbe Bole", "Mayabono Biharini", "Jagorane Jay Bibhabari", "Amar Bhitor O Baahire", "Abar Elo Je Sondhya", "Bojhena Shey Bojhena (Female)", "Cholo Jai", "Khola Haowa", "Mor Bhabonare", "Sakhi Bhabona Kahare", "Tomar Khola Hawa", "Ami Jenehune", "Rabindra Sangeet", "Akash Bhora", "Aji Jhoro Jhoro"],
    "Kishore Kumar": ["Aamar Swapno Tumi", "Asha Chhilo", "Eki Holo", "Noyono Sarasi Keno", "Aaj Ei Din Take", "Chirodini Tumi Je Amar", "Sei Raate Raat Chilo Purnima", "Ami Je Ke Tomar", "O Amar Sajani", "Tobu Bole Keno", "Shing Nei Tobu", "Prithibi Bodle Geche", "Amar Moner Kon", "Kotha Hoyechilo", "Ogo Nirupoma"],
    "Kumar Sanu": ["Tumi Elena", "Priyo Bandhabi", "Sagor Dake", "Tumi Je Amari", "Ei Mon Tomake Dilam", "O Madhuri", "Ami Sei Manus", "Tomake Chai", "Koto Je Sagor", "E Jibon Keno", "Bhalobashi", "Moner Manush", "Swapno", "Ashbo Firiye", "Jibon Saathi"],
    "Manna Dey": ["Coffee Houser Sei Adda", "Ami Je Jalsagharer", "Teer Bindha Pakhi", "Se Amar Chhoto Bon", "Sobai To Sukhi Hote Chay", "Hoyto Tomari Jonno", "Jodi Kagoje Lekho Naam", "Sundari Go", "E Ki Oporup Rupe", "Deep Chhilo", "Ke Tumi", "Poush Toder", "Ami Jamini Tumi", "O Keno Eto Shundori", "Ami Tar Thikana"],
    "Hemanta Mukhopadhyay": ["Ei Raat Tomar Amar", "O Nadire", "Muche Jaoa Dinguli", "Amay Proshno Kore", "Surjodoyar Deshe", "O Akash Sona Sona", "Runner", "Ami Dur Hote Tomarei", "Palki Chole", "Meghe Meghe", "Shonor Tori", "Bhalobese Sakhi", "Tumi Robe Nirobe", "Aguner Poroshmoni", "Klanti"],
    "Rabindra Sangeet": ["Ekla Cholo Re", "Purano Sei Diner Kotha", "Amar Hiyar Majhe", "Jodi Tor Dak Shune", "Bhalobese Sakhi", "Mayabono Biharini", "Aguner Poroshmoni", "Tumi Robe Nirobe", "Pagla Hawar Badol Dine", "Amar Poran Jaha Chay", "Je Raate Mor Duarguli", "Sokhi Bhabona Kahare", "O Je Mane Na Mana", "Chokher Aalo", "Momo Chitte"],
    "Nazrul Geeti": ["Karar Oi Louho Kopat", "O Mon Romzaner Oi Rozar Sheshe", "Durgama Giri Kantara Maru", "Amar Aponar Cheye Apon Je Jon", "Shukno Patar Nupur Paye", "Mora Jhonjhar Moto Uddam", "Bagichay Bulbuli Tui", "Poddar Dheu Re", "Ami Chirotore Dure", "Tora Dekhe Ja", "Khelichho E Biswa Loye", "Rum Jhum Jhum", "Eki Oporup Rupe Maa Tomay", "Jago Nari Jago Bonhi", "Chol Chol Chol"],
    "Folk": ["Bondhu Tor Laiga Re", "Sadher Lau", "Nisha Lagilo Re", "Age Ki Sundor Din Kataitam", "Barir Pashe Arshinagar", "Khachar Bhitor Ochin Pakhi", "Dil Ki Doya Hoy Na", "Shona Bondhu", "Kolom Kotoi", "Komola Sundori", "Gari Chole Na", "Gram Chara Oi Ranga Matir Poth", "Sohag Chand", "Lal Paharir Deshe", "Bokul Phul"]
}

def generate_csv():
    headers = ["language", "youtube_url", "title", "artist", "lyrics"]
    
    # Generate list
    data = []
    
    # Helper
    def add_song(lang, artist, title):
        search_query = f"ytsearch1:{title} - {artist} Official Audio"
        # Sanitize for CSV
        display_title = f"{title} - {artist}"
        data.append({
            "language": lang,
            "youtube_url": search_query,
            "title": display_title,
            "artist": artist,
            "lyrics": ""
        })

    # Add English
    count_eng = 0
    for artist, songs in ENGLISH_ARTISTS.items():
        for song in songs:
            add_song("english", artist, song)
            count_eng += 1
            
    # Add Bangla
    count_bangla = 0
    for artist, songs in BANGLA_ARTISTS.items():
        for song in songs:
            add_song("bangla", artist, song)
            count_bangla += 1
            
    # Fill remaining if needed (should be enough)
    print(f"Generated {count_eng} English and {count_bangla} Bangla songs.")
    
    # Write CSV
    with open("youtube_sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
        
    print("CSV file 'youtube_sources.csv' has been generated successfully.")

if __name__ == "__main__":
    generate_csv()
