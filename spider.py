import gazpacho

url1 = "https://www.bilibili.com/"
bhtml = gazpacho.get(url1)
btree = gazpacho.Soup(bhtml)
bdivs = btree.find("h3", attrs={"class": "bili-video-card__info--tit"})
for eachdiv in bdivs:
    print(eachdiv.find("a").strip())

video_cards = btree.find("picture", attrs={"class": "v-img bili-video-card__cover"}, mode="all")
for card in video_cards:
    cover_img = card.find("img")
    cover_url = cover_img.attrs.get("src")
    print(cover_url)

url2 = "https://news.tongji.edu.cn/tjkx1.htm"
tongjihtml = gazpacho.get(url2)
tongjitree = gazpacho.Soup(tongjihtml)
tongjidivs = tongjitree.find("div", attrs={"class": "tz-txt fr"})
newlist = []
for eachdiv in tongjidivs:
    newlist.append(eachdiv.find("p").strip())
print(newlist)

