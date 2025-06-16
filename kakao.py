import requests
from flask import Blueprint, request, jsonify
import os

KAKAO_REST_API_KEY = os.getenv('KAKAO_REST_API_KEY')
kakao_bp = Blueprint('kakao_bp', __name__)

@kakao_bp.route('/api/kakao-route', methods=['POST'])
def get_kakao_route():
    data = request.get_json()
    origin = data.get("origin", "")
    destination = data.get("destination", "")
    waypoints = data.get("waypoints", "")

    if not origin or not destination:
        return jsonify({"error": "origin/destination is missing"}), 400

    url = "https://apis-navi.kakaomobility.com/v1/directions"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}",
        "Content-Type": "application/json"
    }
    params = {
        "origin": origin,
        "destination": destination,
        "waypoints": waypoints  # waypoints가 없으면 빈 문자열로 전송됨
    }

    res = requests.get(url, headers=headers, params=params)
    if res.status_code != 200:
        return jsonify({
            "error": "Kakao API failed",
            "status_code": res.status_code,
            "details": res.text
        }), 500

    data_json = res.json()
    routes = data_json.get("routes", [])
    if not routes:
        return jsonify({"error": "No routes returned from Kakao"}), 404

    return jsonify(routes[0])
