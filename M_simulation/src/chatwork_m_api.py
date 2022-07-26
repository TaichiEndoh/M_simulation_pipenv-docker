#coding=UTF-8 https://www.chatwork.com/
import requests

def api_chat(mes):
	BASE_URL = ''#https://api.chatwork.com

	#Setting
	roomid   = '' #ルームIDを記載
	message  = mes
	apikey   = '' #APIのKeyを記載

	post_message_url = '{}/rooms/{}/messages'.format(BASE_URL, roomid)

	headers = { 'X-ChatWorkToken': apikey}
	params = { 'body': message }

	r = requests.post(post_message_url,
	                    headers=headers,
	                    params=params)
	print(r)
