import json
import random

# 각 장르에 맞는 영어/한국어 키워드 뱅크 (3배 이상 확장)
keywords = {
    "Dance": {
        "DISCO&Newjack swing": {
            "en": ["Groove", "Night", "Fever", "Magic", "Star", "Funky", "Get Down", "Move", "Boogie", "Electric", "Vibe", "City", "Street", "Rhythm", "Jack", "Swing", "Tonight", "Spotlight", "Glitter", "Starlight", "Rollerblade", "Strobe", "Celebration", "Ecstasy", "Harmony", "Uptown", "Downtown", "Shuffle", "Snap", "Paradise", "Pleasure", "Sunset", "Moonlight", "Affection"],
            "ko": ["그루브", "도시의 밤", "펑키", "리듬", "스윙", "매직", "별빛", "댄스", "움직여", "황홀경", "오늘밤", "조명", "글리터", "축제", "희열", "하모니", "업타운", "다운타운", "셔플", "스냅", "롤러스케이트", "미러볼", "낙원", "파라다이스", "환희", "월광", "애정"]
        },
        "Drum and Bass": {
            "en": ["Velocity", "Break", "Rush", "Jungle", "Dark", "Liquid", "Neuro", "Roller", "Bassline", "Signal", "Echo", "Future", "Tech", "Deep", "Sonic", "Control", "Matrix", "Amen", "Breakbeat", "Sub-bass", "Atmospheric", "Cyborg", "Metropolis", "Vortex", "Dimension", "Parallel", "System", "Reese", "Acceleration", "Gravity", "Code", "Abyss"],
            "ko": ["속도", "정글", "어둠", "미래", "소닉", "베이스라인", "신호", "울림", "깊은", "질주", "매트릭스", "통제", "브레이크비트", "사이보그", "대도시", "차원", "시스템", "평행", "소용돌이", "가속", "무중력", "심해", "코드", "카오스", "패턴"]
        },
        "EDM&IDM": {
            "en": ["Anthem", "Drop", "Pulse", "Glitch", "Synthetic", "Dream", "Atmosphere", "ID", "Digital", "Code", "Abstract", "Logic", "Arc", "Spark", "Ethereal", "Awake", "Vision", "Trance", "Plur", "Utopia", "Dystopia", "Singularity", "Algorithm", "Fractal", "Ascension", "Lucid", "Sequence", "Consciousness", "Virtual", "Reality", "Circuit"],
            "ko": ["축제", "드랍", "신스", "꿈", "디지털", "코드", "불꽃", "논리", "추상", "환상", "각성", "비전", "트랜스", "유토피아", "디스토피아", "알고리즘", "프랙탈", "승천", "자각몽", "시퀀스", "의식", "가상현실", "회로", "네트워크", "초월"]
        },
        "Future Bass&Dubstep": {
            "en": ["Future", "Wobble", "Growl", "Drop", "Heavy", "Space", "Stars", "Neon", "Crystal", "Glitch", "Emotion", "Sky", "Rise", "Fall", "Bass", "Supernova", "Gravity", "Serenity", "Melancholy", "Blade", "Laser", "Hologram", "Impact", "Shatter", "Reverb", "Delay", "Crush", "Tears", "Bloom", "Collapse"],
            "ko": ["미래", "우주", "네온", "크리스탈", "감성", "하늘", "드랍", "베이스", "상승", "하강", "중력", "신세계", "평온", "우울", "홀로그램", "레이저", "충격", "산산조각", "잔향", "딜레이", "파편", "감정의 파동", "눈물", "개화", "붕괴"]
        },
        "House": {
            "en": ["House", "Deep", "Soulful", "Jackin'", "Tech", "Progressive", "Acid", "Groove", "Feeling", "Unity", "Love", "Chicago", "Detroit", "Underground", "Sunrise", "Together", "Balearic", "Ibiza", "Warehouse", "Journey", "Ritual", "Devotion", "Body", "Sweat", "Rapture", "Release", "Freedom", "Dancefloor", "Vocal", "Piano"],
            "ko": ["하우스", "딥", "소울", "그루브", "느낌", "사랑", "언더그라운드", "시카고", "새벽", "우리", "함께", "일출", "이비자", "웨어하우스", "여정", "의식", "헌신", "몸", "땀", "황홀", "해방", "주문", "자유", "단결", "피아노"]
        },
        "Real Music": {
            "en": ["Organic", "Live", "Session", "Jam", "Groove", "Sunset", "Sunrise", "Island", "Celebration", "Human", "Connect", "Flow", "Breeze", "Wave", "Bonfire", "Tribe", "Percussion", "Acoustic", "Unplugged", "Shoreline", "Horizon", "Gathering", "Earth", "Roots", "Vibration", "Pure"],
            "ko": ["라이브", "세션", "그루브", "일몰", "일출", "축하", "연결", "흐름", "자연", "파도", "바람", "인간적인", "모닥불", "부족", "의식", "타악기", "어쿠스틱", "해변", "수평선", "화합", "모임", "진솔한", "뿌리", "대지", "진동"]
        }
    },
    "Hiphop": {
        "Real Music": {
            "en": ["Cypher", "Freestyle", "Boom Bap", "Golden Era", "Flow", "Rhyme", "Street", "Concrete", "Soul", "Jazz", "Vibe", "Reality", "Message", "Chronicle", "Truth", "Asphalt", "Ghetto", "Struggle", "Triumph", "Mic", "Turntable", "Beat", "Lyrical", "Testimony", "Legacy", "Alley", "Life", "Story", "Graffiti"],
            "ko": ["싸이퍼", "프리스타일", "붐뱁", "플로우", "라임", "거리", "현실", "메시지", "재즈", "진심", "진실", "연대기", "목소리", "아스팔트", "투쟁", "승리", "마이크", "턴테이블", "비트", "가사", "증언", "유산", "서사", "골목", "인생", "이야기", "그래피티", "독백"]
        }
    },
    "Pop": {
        "Ballade": {
            "en": ["Memory", "Rain", "Star", "Tear", "Heart", "Whisper", "Promise", "Time", "Faded", "Alone", "Echo", "Love", "Story", "Goodbye", "Yesterday", "Regret", "Silhouette", "Photograph", "Last", "First", "Forever", "Never", "Moonlight", "Solitude", "Confession", "Scars", "Absence", "Longing"],
            "ko": ["기억", "비", "별", "눈물", "마음", "속삭임", "약속", "시간", "사랑 이야기", "이별", "어제", "안녕", "편지", "후회", "실루엣", "사진", "마지막", "처음", "영원", "달빛", "고독", "고백", "못잊어", "그리움", "상처", "부재", "흔적"]
        },
        "City Pop": {
            "en": ["City", "Night", "Drive", "Summer", "Ocean", "Breeze", "Twilight", "Neon", "Magic", "Plastic", "Love", "Weekend", "Coastline", "Sunset", "Freeway", "Midnight", "Riviera", "Resort", "Cocktail", "Expressway", "Cigarette", "Affair", "Rendezvous", "Tropic", "Bay", "Player", "Lights", "Highway", "Melody"],
            "ko": ["도시", "여름밤", "드라이브", "바다", "산들바람", "노을", "네온사인", "주말", "사랑", "추억", "해안선", "자정", "리비에라", "리조트", "칵테일", "고속도로", "담배", "랑데부", "열대", "만", "플레이어", "낭만", "불빛", "하이웨이", "멜로디", "꿈결"]
        },
        "R&B": {
            "en": ["Velvet", "Smooth", "Groove", "Soul", "Late Night", "Mood", "Vibe", "Touch", "Slow Jam", "Desire", "Sensual", "Closer", "Intimate", "Candlelight", "Obsession", "Whiskey", "Silk", "Skin", "Gravity", "Crave", "Temptation", "After Hours", "Rainy Night", "Confessions", "Body Heat", "Red Wine", "Secret"],
            "ko": ["벨벳", "그루브", "소울", "늦은 밤", "무드", "느낌", "터치", "욕망", "더 가까이", "너의 향기", "촛불", "비밀", "집착", "위스키", "실크", "피부", "갈망", "유혹", "심야", "비오는 밤", "고백", "은밀한", "체온", "와인", "단둘이"]
        },
        "Real Music": {
            "en": ["Acoustic", "Unplugged", "Story", "Coffee", "Morning", "Sunday", "Simple", "Smile", "Daydream", "With You", "Letter", "Window", "Diary", "Polaroid", "Kitchen", "Garden", "Autumn", "Pillow", "Conversation", "Promise", "Sweater", "Footprints", "Sunbeam", "Bookshelf", "Childhood"],
            "ko": ["어쿠스틱", "이야기", "커피", "아침", "일요일", "미소", "너와 함께", "백일몽", "편지", "창가에서", "단순함", "일기", "폴라로이드", "주방", "정원", "가을", "베개", "대화", "약속", "스웨터", "발자국", "햇살", "책장", "어린시절"]
        },
        "UK Garage": {
            "en": ["Garage", "2-Step", "Rewind", "Heartbreak", "London", "Underground", "Sweet", "Sunshine", "Weekend", "Moving", "Closer", "Crush", "Vibe", "Champagne", "Sunrise", "M25", "So Solid", "Heartache", "Butterflies", "Flava", "Riddim", "Baseline", "UKG", "Notice Me", "My Boo"],
            "ko": ["개러지", "투스텝", "리와인드", "런던", "주말", "설렘", "햇살", "더 가까이", "심장박동", "첫눈에", "이 느낌", "샴페인", "일출", "고속도로", "두근거림", "상심", "특별한 맛", "리듬", "베이스라인", "런던의 밤", "주말의 끝", "내 사랑", "신호"]
        }
    },
    "Rock": {
        "Folk": {
            "en": ["River", "Mountain", "Road", "Fire", "Home", "Wanderer", "Story", "Echoes", "Woods", "Dust", "Fields", "Simple Song", "Journey", "Traveler", "Creek", "Canyon", "Whiskey", "Harvest", "Roots", "Ghost", "Legend", "Pilgrim", "Testament", "Barefoot", "Homeward", "Train", "Bones"],
            "ko": ["강", "산", "길", "불꽃", "집", "방랑자", "이야기", "메아리", "숲", "들판", "소박한 노래", "여정", "나그네", "계곡", "협곡", "위스키", "수확", "뿌리", "유령", "전설", "순례자", "증표", "맨발", "귀향", "기차", "뼛속"]
        },
        "Real Music": {
            "en": ["Anthem", "Riff", "Rebellion", "Freedom", "Generation", "Echo", "Gasoline", "Electric", "Heartbeat", "Chasing", "Runaway", "Youth", "Rage", "Static", "Stereo", "Vinyl", "Leather", "Riot", "Teenage", "Fist", "Engine", "Vengeance", "Revolution", "Outcast", "Ignite"],
            "ko": ["외침", "리프", "자유", "세대", "메아리", "질주", "심장박동", "도망자", "일렉트릭", "청춘", "분노", "저항", "스테레오", "바이닐", "가죽", "폭동", "십대", "주먹", "엔진", "복수", "아날로그", "혁명", "아웃사이더", "점화"]
        },
        "Rock": {
            "en": ["Midnight", "Ride", "Edge", "Storm", "Rise", "Falling", "Break", "Revolution", "Static", "Scream", "Electric", "City Lights", "Outlaw", "Shadow", "Vulture", "Sirens", "Asylum", "Kingdom", "Throne", "Gasoline", "Neon", "Concrete", "Ruin", "Fade", "Chaos", "Abyss"],
            "ko": ["자정", "질주", "폭풍", "반항", "혁명", "외침", "일렉트릭", "도시의 불빛", "경계선", "그림자", "무법자", "광기", "사이렌", "왕국", "왕좌", "네온", "콘크리트", "폐허", "사라지다", "심연", "절망", "질서", "혼돈", "파멸"]
        }
    }
}


def generate_lang_titles(words, count):
    """한 언어로만 된 고유한 제목을 생성합니다."""
    titles = set()

    # 단어 수가 충분한지 확인
    can_combine_three = len(words) >= 3
    can_combine_two = len(words) >= 2

    while len(titles) < count:
        num_words = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]

        try:
            if num_words == 3 and can_combine_three:
                title = " ".join(random.sample(words, 3))
            elif num_words == 2 and can_combine_two:
                title = " ".join(random.sample(words, 2))
            else:
                title = random.choice(words)

            titles.add(title.title()) # 'Title Case'로 통일성 부여
        except IndexError:
            # 혹시 모를 경우를 대비 (단어 풀이 매우 작을 때)
            continue

    return list(titles)

# --- 메인 실행 로직 ---
final_json = {}
TOTAL_TITLES_PER_GENRE = 2000
EN_COUNT = TOTAL_TITLES_PER_GENRE // 2
KO_COUNT = TOTAL_TITLES_PER_GENRE - EN_COUNT

for main_genre, sub_genres in keywords.items():
    final_json[main_genre] = {}
    for sub_genre, words in sub_genres.items():
        print(f"Generating titles for {main_genre} -> {sub_genre}...")

        # 영어 제목 생성
        en_titles = generate_lang_titles(words["en"], EN_COUNT)

        # 한국어 제목 생성
        ko_titles = generate_lang_titles(words["ko"], KO_COUNT)

        # 합친 후 랜덤으로 섞기
        combined_list = en_titles + ko_titles
        random.shuffle(combined_list)

        final_json[main_genre][sub_genre] = combined_list

# JSON 파일로 저장
file_path = "music_titles.json"
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(final_json, f, ensure_ascii=False, indent=2)

print(f"\nSuccessfully generated {file_path} with {TOTAL_TITLES_PER_GENRE} unique titles per sub-genre.")
print("Keyword banks have been expanded significantly for greater variety.")