"""Original, explicitly segmented everyday prose; no story corpus or runtime catalog.

Slots are narrowly typed by the declared sentence family. Readings and POS come
only from the checksummed paired dictionary. These are authored training rules,
not an independent human language review or a constraint used during inference.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from scripts.lexical_rules import lexical_collision
from scripts.semantic_keys import core_key


POOLS = {
    'person': '父 母 兄 姉 妹 祖父 祖母 友達',
    'drink': 'お茶 紅茶 緑茶 麦茶 ほうじ茶 コーヒー ココア 牛乳 豆乳 ジュース',
    'food': 'ご飯 パン お粥 カレー 味噌汁 スープ 煮物 焼き魚 サラダ クッキー ケーキ',
    'produce': '人参 大根 玉ねぎ じゃがいも キャベツ 白菜 胡瓜 トマト 茄子 蓮根',
    'fruit': '林檎 蜜柑 苺 葡萄 梨 桃 西瓜 メロン オレンジ',
    'clothes': 'シャツ タオル ハンカチ セーター マフラー 手袋 靴下 上着 帽子',
    'book': '小説 短編集 随筆 詩集 伝記 図鑑 雑誌 旅行記 教科書 絵本',
    'document': '資料 報告書 書類 設計図 案内文 説明文 記事 台本 説明書',
    'plan': '予定 計画 日程 時間 手順 方法 方針 条件 目的',
    'note': 'ノート 手帳 メモ帳 日記 読書帳 記録帳',
    'container': '鞄 リュック 手提げ 袋 箱',
    'stationery': '鉛筆 ペン ノート 手帳 メモ帳 色鉛筆 クレヨン',
    'plant': 'ひまわり 朝顔 薔薇 紫陽花 菫 百合 桜 梅',
    'scene': '山 海 川 空 雲 花 木 橋 家 車 船 犬 猫 鳥 馬 風景',
    'instrument': 'ピアノ ギター バイオリン トランペット ハープ 笛 琴 太鼓',
    'subject': '算数 理科 漢字 ひらがな 歴史',
    'sport': 'テニス 卓球 バドミントン 水泳 剣道 柔道 バスケットボール バレー',
    'game': '将棋 囲碁 チェス オセロ かるた トランプ 双六 パズル',
    'destination': '図書館 博物館 美術館 科学館 動物園 植物園 水族館 公民館 公園 市場',
    'transport': 'バス 電車 路面電車 汽車 タクシー 自転車',
    'path': '並木道 遊歩道 川沿い 商店街 参道 坂道 山道 湖畔 海辺 河原',
    'room': '部屋 居間 寝室 書斎 教室 台所',
    'surface': '机 棚 床 窓 扉 戸 手すり',
    'tool': '道具 部品 ねじ 刷毛 ブラシ 紙やすり 布 スポンジ',
    'material': '紙 布 毛糸 陶土 木綿 絹 フェルト 帆布',
    'measure': '温度 湿度 長さ 重さ 速さ 面積 体積 密度 圧力',
    'observation': '花 葉 種 昆虫 蟻 蝉 蜻蛉 植木 苗',
    'bird': '雀 燕 鴨 鷺 鶴 鳩 鶯 白鳥',
}
# Topic, family, pattern, family-specific slot domains. Multiple patterns express
# different relationships, rather than expanding a sentence with time prefixes.
FAMILIES = [
 ('shopping','colored_items','{color} {item} を {action}',{'color':'白い 赤い 青い 黄色い 黒い','item':'鞄 靴 傘 箱 皿 瓶 缶 帽子 上着 椅子 机 棚 布 紙 ペン 鉛筆 手袋 シャツ ボタン タオル','action':'買う 選ぶ 探す'}),
 ('art','colored_drawing','{color} {item} を {action}',{'color':'白い 赤い 青い 黄色い 黒い','item':'鞄 靴 傘 箱 皿 瓶 缶 帽子 上着 椅子 机 棚 布 紙 ペン 鉛筆 手袋 シャツ ボタン タオル','action':'描く 描いた 描きます'}),
 ('home','colored_clothes','{color} {clothes} を {action}',{'color':'白い 赤い 青い 黄色い 黒い','action':'洗う 畳む 干す'}),
 ('food','meal_together','{person} と {place} で {food} を {action}',{'place':'家 居間 台所 食堂 庭 公園','action':'食べる 食べた'}),
 ('food','drink_together','{person} と {place} で {drink} を {action}',{'place':'家 居間 台所 食堂 庭 喫茶店','action':'飲む 飲んだ'}),
 ('cooking','recipe_together','{person} に 教わった {food} を {partner} と 作る',{'partner':POOLS['person']}),
 ('home','wash_together','{person} と {clothes} を 洗って {storage} に しまう',{'storage':'棚 戸棚 引き出し 箱'}),
 ('home','choose_for','{person} に 贈る {item} を {shop} で 選ぶ',{'item':'鞄 マフラー ハンカチ 帽子 手袋 シャツ セーター','shop':'店 雑貨屋 商店街 市場'}),
 ('learning','learn_together','{person} と {place} で {subject} を {action}',{'place':'家 教室 学校 図書館 図書室 公民館','action':'学ぶ 勉強する'}),
 ('books','read_together','{person} と {place} で {book} を {action}',{'place':'家 居間 教室 学校 図書館 図書室 公民館','action':'読む 読んだ'}),
 ('books','borrow_for','{person} に 見せる {book} を {place} で 選ぶ',{'place':'本屋 古本屋 図書館 図書室'}),
 ('work','explain_to','{person} に {meeting} で {document} の {aspect} を 説明する',{'meeting':'会議 打ち合わせ 研修 勉強会','aspect':'目的 読み方 書き方'}),
 ('work','check_together','{person} と {place} で {document} を {action}',{'place':'家 教室 学校 公民館 作業場','action':'読む 読み直す 見直す 確認する まとめる'}),
 ('work','notes_in_meeting','{meeting} で {activity} の {plan} を {note} に 書き留める',{'meeting':'会議 打ち合わせ 研修 勉強会','activity':'作業 制作 開発 設計 調査'}),
 ('science','record_together','{person} と {place} で {observation} の 様子 を {recording} で 記録する',{'place':'家 教室 学校 公園 庭','recording':'写真 動画 図'}),
 ('travel','walk_to','{person} と {path} を 歩いて {destination} へ 行く',{}),
 ('travel','trip_together','{person} と {transport} で {destination} へ {action}',{'action':'行く 向かう 出かける'}),
 ('social','show_drawing','{person} に 見せる ために {object} の 絵 を {medium} で 描く',{'object':'家 花 犬 猫 船 山 海 空 木 橋 鳥 雲 街 川 池 島 馬 象 車','medium':'色鉛筆 クレヨン 木炭 筆'}),
 ('art','draw_together','{person} と {place} で {object} の 絵 を {medium} で 描く',{'place':'家 教室 学校 公園 庭','object':'家 花 犬 猫 船 山 海 空 木 橋 鳥 雲 街 川 池 島 馬 象 車','medium':'色鉛筆 クレヨン 木炭 筆'}),
 ('music','practice_together','{person} と {place} で {instrument} を 練習する',{'place':'家 音楽室 練習室 スタジオ 教室 学校'}),
 ('sport','match_together','{person} と {place} で {sport} の 試合 を 観る',{'place':'体育館 学校 公民館'}),
 ('game','game_together','{person} と {place} で {game} を 楽しむ',{'place':'家 居間 教室 学校 公民館 庭'}),
 ('making','repair_together','{person} と {place} で {item} を 直す',{'place':'家 教室 作業場 作業台','item':'椅子 机 棚 本棚 玩具 模型 鞄'}),
 ('garden','plant_together','{person} と {place} に {plant} の 苗 を 植える',{'place':'庭 花壇 植木鉢'}),
 ('nature','photograph_together','{person} と {place} で {object} の 写真 を 撮る',{'place':'公園 庭 海辺 湖畔 河原 森 丘','object':'花 木 鳥 雲 橋 船 猫 犬'}),
 ('cooking','prepare_food','{person} と {food} を 作る',{}),
 ('cooking','cut_produce','{produce} を {cut} 切る',{'cut':'薄く 細かく 小さく'}),
 ('cooking','wash_cut','{produce} を よく 洗って から 小さく 切る',{}),
 ('cooking','learn_recipe','{person} に 教わった {food} の 作り方 を {note} に 書き残す',{}),
 ('cooking','cook_for','{person} の ために {food} を 作る',{}),
 ('cooking','serve_food','{food} を {dish} に 盛る',{'dish':'皿 大皿 小皿 器'}),
 ('food','eat','{food} を {eat}',{'eat':'食べる 食べた'}),
 ('food','fruit','{fruit} を {action}',{'action':'食べる 食べた 買う 洗う'}),
 ('food','drink','{drink} を {drink_action}',{'drink_action':'飲む 飲んだ'}),
 ('food','tea_for','{person} に {drink} を {serve}',{'serve':'用意する 入れる'}),
 ('food','take_drink','{drink} を {vessel} に 入れる',{'vessel':'水筒 コップ 湯飲み'}),
 ('home','wash','{clothes} を {wash}',{'wash':'洗う 洗った 畳む 畳んだ'}),
 ('home','clean','{surface} を {manner} 拭く',{'manner':'丁寧に きちんと よく'}),
 ('home','store_clothes','{clothes} を 畳んで {storage} に しまう',{'storage':'棚 戸棚 引き出し'}),
 ('home','clothes_gift','{person} に もらった {clothes} を 大切に 使う',{}),
 ('home','pack','{container} に {stationery} を 入れる',{}),
 ('home','room_tidy','{room} の {furniture} を 片付ける',{'furniture':'机 棚 本棚'}),
 ('home','choose_clothes','{person} と {clothes} を 選ぶ',{}),
 ('everyday','prepare_bag','{destination} へ 行く ために {container} を 用意する',{}),
 ('everyday','write_note','{plan} を {note} に 書く',{}),
 ('everyday','put_stationery','{stationery} を {where} に 置く',{'where':'机 棚 箱'}),
 ('everyday','calendar','{person} と {plan} を 確認する',{}),
 ('books','read_book','{book} を {read}',{'read':'読む 読んだ 読みます 読み直す'}),
 ('books','borrowed','{destination} で 借りた {book} を 家 で 読む',{'destination':'図書館 図書室 学校'}),
 ('books','impression','{book} を 読んで 感想 を {note} に 書く',{}),
 ('books','choose_book','{person} と {shop} で {book} を 選ぶ',{'shop':'本屋 古本屋 図書館'}),
 ('learning','study','{person} と {subject} を 勉強する',{}),
 ('learning','lookup','{lookup} を {reference} で 調べる',{'lookup':'漢字 意味 読み方 書き方 名前','reference':'辞書 図鑑 教科書'}),
 ('learning','ask','{person} に {question} を 聞く',{'question':'意味 読み方 書き方 理由 方法 考え方'}),
 ('learning','study_note','{subject} を 学ぶ ために {note} を 用意する',{}),
 ('work','check_document','{document} を {check}',{'check':'読む 読み直す 見直す 確認する まとめる'}),
 ('work','share','{meeting} で {document} を 共有する',{'meeting':'会議 打ち合わせ 研修 勉強会'}),
 ('work','agree','{meeting} で {plan} を 決める',{'meeting':'会議 打ち合わせ 交渉 相談'}),
 ('work','work_notes','{activity} の {plan} を {note} に 書き留める',{'activity':'作業 制作 開発 設計 調査'}),
 ('work','organize','{document} を {manner} 整理する',{'manner':'丁寧に きちんと 手早く'}),
 ('work','explain','{person} に {document} の {aspect} を 説明する',{'aspect':'目的 読み方 書き方'}),
 ('science','measure','{measure} の 変化 を {device} で 調べる',{'device':'装置 実験'}),
 ('science','record','{observation} の 様子 を {recording} で 記録する',{'recording':'写真 動画 図'}),
 ('science','compare','{measure} の 測定 を {place} で 行う',{'place':'教室 作業場 学校'}),
 ('science','observe_note','{observation} を 観察する ために {note} を 用意する',{}),
 ('thinking','examine','{thing} の {aspect} を 調べる',{'thing':'機械 装置 発電機 印刷機 モーター ポンプ','aspect':'仕組み 共通点 変化'}),
 ('thinking','review','{activity} の {aspect} を 見直す',{'activity':'作業 制作 開発 設計 調査','aspect':'方法 手順 条件 目的 根拠'}),
 ('thinking','organize_thought','{activity} の {aspect} を {note} に まとめる',{'activity':'実験 調査 観察 研究','aspect':'結果 予想 仮説 疑問'}),
 ('shopping','buy','{person} と {shop} で {item} を 買う',{'shop':'店 雑貨屋 市場','item':'鞄 傘 タオル ハンカチ 帽子 手袋'}),
 ('shopping','grocery','{person} と {produce} を 買う',{}),
 ('shopping','bookshop','{person} と {shop} で {stationery} を 選ぶ',{'shop':'店 雑貨屋 文房具屋'}),
 ('shopping','gift','{person} の 誕生日 に {gift} を 選ぶ',{'gift':'鞄 マフラー ハンカチ 帽子 手袋 本'}),
 ('travel','go','{transport} で {destination} へ 行く',{}),
 ('travel','map','{map} で {destination} まで の 道 を 確認する',{'map':'地図 案内図 観光地図'}),
 ('travel','plan_trip','{person} と {destination} へ 行く 計画 を 決める',{}),
 ('travel','route','{map} で {destination} まで の 道 を 調べる',{'map':'地図 案内図 観光地図 道路地図'}),
 ('town','walk','{person} と {path} を 歩く',{}),
 ('town','wait','{person} と {place} で 待ち合わせて {destination} へ 向かう',{'place':'駅 駅前 バス停'}),
 ('town','notice','{event} の 案内 を {where} に 出す',{'event':'交流会 演奏会 展示会 読書会 勉強会','where':'掲示板 受付'}),
 ('social','promise','{person} と {destination} で 会う 約束 を した',{}),
 ('social','help','{person} の {activity} を 手伝う',{'activity':'料理 掃除 片付け 準備 作業'}),
 ('social','show','{person} に {item} を 見せる',{'item':'写真 絵 地図 作品 手紙 ノート'}),
 ('social','deliver','{person} に {item} を 届ける',{'item':'手紙 鞄 書類 資料 本 荷物'}),
 ('art','draw','{object} を {draw}',{'object':'家 花 犬 猫 船 山 海 空 星 月 木 橋 鳥 雲 街 川 池 島 馬 象 車 椅子 鞄 靴','draw':'描く 描いた 描きます'}),
 ('art','picture','{object} の 絵 を {medium} で 描く',{'object':'家 花 犬 猫 船 山 海 空 木 橋 鳥 雲 街 川 池 島 馬 象 車','medium':'色鉛筆 クレヨン 木炭 筆'}),
 ('art','display','{person} が 描いた {art} を 部屋 に 飾る',{'art':'絵 水彩画 油絵 素描'}),
 ('art','view','{person} と {venue} で {art} を 眺める',{'venue':'美術館 展示室 展示会','art':'絵画 彫刻 水彩画 油絵 版画 写真'}),
 ('music','practice','{person} と {instrument} を 練習する',{}),
 ('music','listen','{instrument} の 演奏 を {place} で 聴く',{'place':'ホール 音楽室 劇場 公会堂'}),
 ('music','prepare','{instrument} を 練習する ために {place} へ 行く',{'place':'音楽室 練習室 スタジオ'}),
 ('sport','practice','{person} と {sport} を 練習する',{}),
 ('sport','watch','{sport} の 試合 を {place} で 観る',{'place':'体育館 テレビ 学校'}),
 ('sport','prepare','{sport} の 試合 を 観る ために {container} を 用意する',{}),
 ('game','play','{person} と {game} を 楽しむ',{}),
 ('game','rules','{game} の 規則 を {reference} で 調べる',{'reference':'本 説明書 図鑑'}),
 ('game','explain','{person} に {game} の 規則 を 説明する',{}),
 ('making','fix','{person} と {item} を 直す',{'item':'椅子 机 棚 本棚 玩具 模型 鞄'}),
 ('making','tools','{item} を 直す ために {tool} を 用意する',{'item':'椅子 机 棚 本棚 玩具 模型 鞄'}),
 ('making','model','{person} と {object} の 模型 を 作る',{'object':'家 船 車 飛行機 橋'}),
 ('craft','choose','{person} と {material} を 選ぶ',{}),
 ('craft','fold','{color} 紙 で {shape} を 折る',{'color':'赤い 青い 白い 黄色い','shape':'鶴 船 箱'}),
 ('craft','sew','{person} と {cloth} で {item} を 作る',{'cloth':'木綿 麻 帆布 フェルト','item':'鞄 袋 手提げ'}),
 ('exercise','stretch','{part} を {manner} 伸ばす',{'part':'腕 脚 背中','manner':'ゆっくり 気持ちよく 少しずつ'}),
 ('exercise','walk','{person} と {path} を ゆっくり 歩く',{}),
 ('health','rest','{room} で {manner} 休む',{'manner':'少し ゆっくり 静かに'}),
 ('health','wash','{person} と 手 を よく 洗う',{}),
 ('care','prepare','{person} に {item} を 用意する',{'item':'タオル ハンカチ 上着 帽子 毛糸'}),
 ('care','help','{person} の {activity} を 手伝う',{'activity':'掃除 片付け 準備 手入れ'}),
 ('garden','water','{plant} に {manner} 水 を やる',{'manner':'丁寧に 少しずつ ゆっくり'}),
 ('garden','observe','{plant} の {part} を 観察する',{'part':'葉 芽 種 花'}),
 ('garden','plant','{person} と {plant} の 苗 を 植える',{}),
 ('nature','branch','{tree} の 枝 が 風 に 揺れる',{'tree':'桜 梅 松 柳 竹 銀杏 楓'}),
 ('nature','flower','{where} に {plant} の 花 が 咲く',{'where':'庭 花壇 公園 道端'}),
 ('nature','photo','{person} と {where} の 写真 を 撮る',{'where':'山 海 川 湖 花壇 公園 庭 橋 森 丘'}),
 ('animals','bird','{bird} が {where} を 飛ぶ',{'where':'空 川 海 湖 池'}),
 ('animals','observe','{person} と {animal} を 観察する',{'animal':'猫 犬 蝶 蟻 鳥 蜻蛉 蝉 かぶと虫 金魚'}),
 ('senses','listen','{where} で {sound} の 音 を 聞く',{'where':'部屋 窓辺 庭 公園','sound':'風 雨 鐘 風鈴'}),
 ('senses','texture','{first} と {second} の 手触り を 比べる',{'first':'木綿 麻 絹','second':'木綿 麻 絹 フェルト 帆布'}),
 ('weather','prepare','雨 が 降って いる ので {item} を 用意する',{'item':'傘 折り畳み傘 長靴 上着'}),
 ('weather','cloud','{cloud} 雲 が {direction} へ 流れる',{'cloud':'白い 大きな 小さな','direction':'東 西 南 北'}),
 ('season','look','{season} の {plant} の 様子 を 観察する',{'season':'春 夏 秋 冬'}),
 ('season','prepare','{season} の {event} の 準備 を 手伝う',{'season':'春 夏 秋 冬','event':'展示会 演奏会 交流会 読書会'}),
]


def prepare(out):
    base = Path('data/paired-v1')
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in {**manifest['split_sha256'], 'vocab.json': manifest['vocab_sha256']}.items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'paired source hash mismatch: {name}')
    vocab_bytes = (base / 'vocab.json').read_bytes()
    vocab = json.loads(vocab_bytes)
    by_surface = defaultdict(list)
    for token in vocab['tokens'][7:]:
        by_surface[token['surface']].append(token)
    def token(surface):
        matches = by_surface[surface]
        if len(matches) != 1:
            raise ValueError(f'lexeme must be explicit and unambiguous: {surface}: {matches}')
        return matches[0]
    parts = {s: [json.loads(line) for line in (base / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()]
             for s in ('train', 'validation', 'test', 'family-test')}
    for rows in parts.values():
        for row in rows:
            row['training_group'] = 'paired'
    seen = {core_key(r['text']) for rows in parts.values() for r in rows}
    stats = Counter()
    family_counts = Counter()
    for topic, family, pattern, custom in FAMILIES:
        names = list(dict.fromkeys(word[1:-1] for word in pattern.split() if word.startswith('{')))
        pools = [custom.get(name, POOLS.get(name, '')).split() for name in names]
        if any(not pool for pool in pools):
            raise ValueError(f'missing slot pool: {family}')
        # Validate every slot and literal even when length constraints reject it.
        for pool in pools:
            for value in pool:
                token(value)
        for word in pattern.split():
            if not word.startswith('{'):
                token(word)
        for values in itertools.product(*pools):
            if len(set(values)) != len(values):
                stats['repeated_slot_value'] += 1
                continue
            slots = dict(zip(names, values))
            words = [slots[w[1:-1]] if w.startswith('{') else w for w in pattern.split()]
            tokens = [token(w) for w in words] + [token('。')]
            kana = ''.join(t['kana'] for t in tokens)
            text = ''.join(t['surface'] for t in tokens)
            if not 5 <= len(kana) <= 40 or len(tokens) + 3 > 64:
                stats['length'] += 1
                continue
            if any(lexical_collision(a, b) for a, b in zip(tokens, tokens[1:])):
                stats['boundary'] += 1
                continue
            key = core_key(text)
            if key in seen:
                stats['duplicate_or_protected'] += 1
                continue
            seen.add(key)
            bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
            # Keep most valid combinations unseen. Exhaustively teaching the
            # small Cartesian domain rewards memorization instead of transfer.
            if bucket >= 30:
                stats['reserved_unseen_combination'] += 1
                continue
            split = 'validation' if bucket >= 25 else 'test' if bucket >= 20 else 'train'
            d = 3 if len(kana) <= 12 else 4 if len(kana) <= 22 else 5
            row = dict(id='everyday-v3-' + hashlib.sha256(text.encode()).hexdigest()[:20], text=text, kana=kana,
                       segments=[{k: t[k] for k in ('surface','kana','pos')} for t in tokens],
                       ids=[0, d, *[t['id'] for t in tokens], 1], source='original-agent-authored',
                       license='CC0-1.0', trust='auto_aligned', training_group='everyday',
                       generation=dict(topic=topic, family='everyday-v3-' + family,
                                       method='authored typed slots; no runtime sentence constraints', slots=slots))
            parts[split].append(row)
            family_counts[topic + '/' + family] += 1
    out.mkdir(parents=True, exist_ok=True)
    (out / 'vocab.json').write_bytes(vocab_bytes)
    result = dict(description='Original everyday Japanese prose; no SimpleStories', license='CC0-1.0',
                  source=dict(dataset='original-everyday-v3', license='CC0-1.0', authors=['Codex; independent human review pending']),
                  parent_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
                  vocab_sha256=hashlib.sha256(vocab_bytes).hexdigest(), split_sha256={}, splits={},
                  source_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  split_strategy='Existing paired splits protected; new normalized bodies SHA256 buckets: 20% train, 5% validation, 5% test, 70% reserved and not exported. Same core never crosses splits. Legacy family-test retained; not proof of unseen syntax.',
                  family_counts=dict(family_counts), stats=dict(stats), independent_human_review='pending')
    for split, rows in parts.items():
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        (out / f'{split}.jsonl').write_bytes(payload)
        result['split_sha256'][f'{split}.jsonl'] = hashlib.sha256(payload).hexdigest()
        result['splits'][split] = len(rows)
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(splits=result['splits'], stats=dict(stats), families=len(family_counts))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('build/everyday-v3-data'))
    prepare(parser.parse_args().out_dir)
