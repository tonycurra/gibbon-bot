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
        self.assertIn("fw old", body)

    def test_core_blocks_raise_until_fw(self):
        from pathlib import Path
        js = (Path(__file__).resolve().parent / "static/console/js/core.js").read_text()
        self.assertIn("raise blocked", js)
        self.assertIn("info fw=", js)
        self.assertIn("stream 180", js)
        self.assertNotIn('await send("bench", true)', js)
        self.assertNotIn('await send("gpio", true)', js)
        self.assertNotIn('await send("who", true)', js)
