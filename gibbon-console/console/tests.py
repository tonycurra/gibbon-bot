from django.test import TestCase


class HatPageTests(TestCase):
    def test_hat_page_renders_gold_captures(self):
        r = self.client.get("/hat/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn("guess=usb-no-battery", body)
        self.assertIn("guess=usb-battery-on", body)
        self.assertIn("found=0/7", body)
        self.assertIn("found=7/7", body)
        self.assertIn("get id=11 pos=1900", body)
        self.assertIn("batt pack=11.4V", body)
        self.assertIn("44:1d:64:86:3f:54", body)
        self.assertIn("ST3215HS", body)
        self.assertIn("40 ms", body)
        self.assertIn("unwrap", body)
        self.assertIn("memory table", body.lower())

    def test_raw_hand_page(self):
        r = self.client.get("/raw/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn("Hand only", body)
        self.assertIn("light touch", body)
        self.assertIn("only a jam", body)
        self.assertIn("effort", body.lower())
        self.assertIn('id="charts"', body)
        self.assertIn("dump bus bytes", body)
        self.assertNotIn("sensitivity", body.lower())

    def test_home_has_fw_chip(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn('id="fwEl"', body)
        self.assertIn('id="usbEl"', body)
        self.assertIn('id="bleEl"', body)
        self.assertNotIn('id="btnFreeAll"', body)
        self.assertIn('id="btnLockPose"', body)
        self.assertIn("LOCK", body)
        self.assertIn("STOP", body)
        self.assertNotIn("FREE ALL", body)
        self.assertNotIn("trialChip", body)
        self.assertIn("calibration", body)
        self.assertIn("/imitation/", body)
        self.assertIn('"slug": "hand"', body)

    def test_monitor_page(self):
        r = self.client.get("/monitor/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn("Web Bluetooth", body)
        self.assertIn("g1", body)
        self.assertIn("Unplug USB", body)
        self.assertIn("VIN", body)
        self.assertIn("step/s", body)

    def test_imitation_page(self):
        from pathlib import Path
        r = self.client.get("/imitation/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn('id="btnRec"', body)
        self.assertIn('id="btnPlay"', body)
        self.assertIn('id="sense"', body)
        self.assertIn("light touch", body)
        self.assertIn("only a jam", body)
        self.assertIn("STOP if a joint pushes this hard", body)
        self.assertIn("raise to", body)
        js = (Path(__file__).resolve().parent / "static/console/js/imitate.js").read_text()
        self.assertIn("hopWalk", js)
        self.assertIn("guard load", js)
        self.assertIn("guard dload", js)
        self.assertIn("cmdRaiseTo", js)
        self.assertIn('send("estop")', js)
        self.assertIn("WRAP_ABORT", js)
        self.assertNotIn("id 12 to", js)

    def test_legacy_pages_redirect(self):
        r = self.client.get("/graphs/")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/raw/")
        r = self.client.get("/act/")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/")
        r = self.client.get("/calibrate/")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/calibration/")
        r = self.client.get("/calibrate/hand/")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/calibration/hand/")

    def test_calibration_pages(self):
        from pathlib import Path
        slugs = ["hand", "wrist-roll", "wrist-pitch", "elbow", "shoulder-pitch", "yaw"]
        r = self.client.get("/calibration/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        for slug in slugs:
            self.assertIn("/calibration/%s/" % slug, body)
        self.assertIn("class=\"set home\"", body)
        self.assertIn("class=\"set min\"", body)
        self.assertIn("class=\"test\"", body)
        self.assertIn("now°", body)
        self.assertIn("class=\"pos\"", body)
        self.assertIn("class=\"un\"", body)
        r = self.client.get("/calibration/hand/")
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn("closed = 0°", body)
        self.assertIn('id="btnTest"', body)
        self.assertIn('id="inGo"', body)
        self.assertNotIn('id="inMin"', body)
        self.assertNotIn('id="btnZero"', body)
        self.assertIn("/calibration/wrist-roll/", body)
        for slug in slugs:
            p = self.client.get("/calibration/%s/" % slug)
            self.assertEqual(p.status_code, 200, slug)
            self.assertIn('id="btnTest"', p.content.decode())
            self.assertIn("class=\"set home\"", p.content.decode())
        js = (Path(__file__).resolve().parent / "static/console/js/calibrate.js").read_text()
        self.assertIn("TEST_MS", js)
        self.assertIn("rawFromDelta", js)
        self.assertIn("isHand(id)", js)
        self.assertIn("startEdit", js)
        self.assertIn("runTest", js)
        self.assertIn("playPath", js)
        self.assertIn("cmdRaiseTo", js)
        self.assertNotIn("handStep", js)

        p = self.client.get("/calibration/shoulder-pitch/")
        body = p.content.decode()
        self.assertIn("btnRec", body)
        self.assertIn("btnPlay", body)
        self.assertIn('id="meter-12"', body)
        self.assertIn('id="meter-13"', body)
        self.assertIn('id="fightEl"', body)
        self.assertIn("paintFight", js)
        self.assertIn("pairGuess", js)
        self.assertIn("hopToward", js)
        self.assertIn("writeChipWalls", js)
        self.assertIn('aria-label="set this pose as min"', body)
        self.assertNotIn("stepPair", body)
        self.assertNotIn("use now as min", body)
        hand = self.client.get("/calibration/hand/")
        self.assertNotIn("btnRec", hand.content.decode())
        self.assertNotIn('id="meter-12"', hand.content.decode())
        self.assertIn("closed = 0°", hand.content.decode())

    def test_calibration_snapshot_api(self):
        r = self.client.post(
            "/api/calibrations",
            data='{"board":"gibbon-3f54","fw":"20260912","via":"usb","joints":{"17":{"home":1000,"amin":0,"amax":400}}}',
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("ok"))

    def test_motors_redirects_to_calibration(self):
        r = self.client.get("/motors/")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "/calibration/")

    def test_flash_page(self):
        r = self.client.get("/flash/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Web Serial", r.content.decode())

    def test_core_blocks_raise_until_fw(self):
        from pathlib import Path
        js = (Path(__file__).resolve().parent / "static/console/js/core.js").read_text()
        self.assertIn("raise blocked", js)
        self.assertIn("info fw=", js)
        self.assertIn("lockPose", js)
        self.assertIn("window.lockPose", js)
        self.assertIn("[10.8, 20]", js)
        self.assertIn("[9.0, 0]", js)
        self.assertNotIn("LIPO3S_EMPTY", js)
        self.assertNotIn('await send("bench", true)', js)
        self.assertIn('await send("get", true)', js)
        self.assertIn("if (parsed.un == null && prev && prev.un != null)", js)
